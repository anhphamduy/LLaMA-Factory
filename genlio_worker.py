#!/usr/bin/env python3
"""
Genlio LLaMA-Factory Worker

Single worker that handles both fine-tuning and inference jobs:
- Single worker_id per machine
- Single status reporter (no duplicate metrics)
- Polls both finetuning_jobs and inference_jobs tables
- Resource-aware job scheduling (checks GPU memory before accepting)
- Subprocess isolation for training jobs

Usage:
    python genlio_worker.py --database-url "postgresql://..." --poll-interval 5

Environment variables:
    DATABASE_URL: PostgreSQL connection string
    WORKER_ID: Unique identifier for this worker (auto-generated if not set)
    WORKER_NAME: Human-readable name for this worker (optional)
    POLL_INTERVAL: Seconds between polls (default: 5)
    DATA_DIR: Directory for LLaMA-Factory datasets (default: ./data)
    OUTPUT_DIR: Base directory for training outputs (default: ./outputs)
    SUPABASE_URL: Supabase project URL
    SUPABASE_KEY: Supabase service key
    MODEL_CACHE_DIR: Directory to cache downloaded adapters (default: ./model_cache)
    TRAINING_MIN_GPU_GB: Minimum GPU memory for training jobs (default: 12)
    INFERENCE_MIN_GPU_GB: Minimum GPU memory for inference jobs (default: 6)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import platform
import signal
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import psutil
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from transformers import TrainerCallback

# Add LLaMA-Factory to path
LLAMA_FACTORY_ROOT = Path(__file__).parent
sys.path.insert(0, str(LLAMA_FACTORY_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("genlio_worker")

STORAGE_BUCKET = "datasets"
ADAPTERS_BUCKET = "adapters"

Base = declarative_base()


# =============================================================================
# Database Models
# =============================================================================

class FineTuningJob(Base):
    """Mirror of the Genlio FineTuningJob model."""
    __tablename__ = "finetuning_jobs"

    job_id = Column(String(255), primary_key=True)
    job_name = Column(String(255), nullable=True)
    status = Column(String(32), nullable=False, default="pending")
    dataset_id = Column(String(255), nullable=True)
    version = Column(String(255), nullable=True)
    base_model = Column(String(512), nullable=False)
    finetuning_type = Column(String(32), nullable=False, default="lora")
    stage = Column(String(32), nullable=False, default="sft")
    training_config = Column(JSON, nullable=True)
    lora_config = Column(JSON, nullable=True)
    dataset_config = Column(JSON, nullable=True)
    output_dir = Column(Text, nullable=True)
    adapter_path = Column(Text, nullable=True)
    current_step = Column(Integer, nullable=True)
    total_steps = Column(Integer, nullable=True)
    current_epoch = Column(Float, nullable=True)
    total_epochs = Column(Float, nullable=True)
    metrics = Column(JSON, nullable=True)
    created_at = Column(String(64), nullable=False)
    updated_at = Column(String(64), nullable=False)
    started_at = Column(String(64), nullable=True)
    completed_at = Column(String(64), nullable=True)
    error = Column(Text, nullable=True)
    worker_id = Column(String(255), nullable=True)


class InferenceJob(Base):
    """Mirror of the Genlio InferenceJob model."""
    __tablename__ = "inference_jobs"

    job_id = Column(String(255), primary_key=True)
    job_name = Column(String(255), nullable=True)
    status = Column(String(32), nullable=False, default="pending")
    finetuning_job_id = Column(String(255), nullable=True)
    base_model = Column(String(512), nullable=True)
    adapter_path = Column(Text, nullable=True)
    template = Column(String(64), nullable=True)
    inference_mode = Column(String(32), nullable=False, default="single")
    messages = Column(JSON, nullable=True)
    system_prompt = Column(Text, nullable=True)
    test_dataset_id = Column(String(255), nullable=True)
    test_version = Column(String(255), nullable=True)
    prompt_column = Column(String(255), nullable=True)
    expected_column = Column(String(255), nullable=True)
    generation_config = Column(JSON, nullable=True)
    output = Column(JSON, nullable=True)
    rows_processed = Column(Integer, nullable=True)
    rows_total = Column(Integer, nullable=True)
    created_at = Column(String(64), nullable=False)
    updated_at = Column(String(64), nullable=False)
    started_at = Column(String(64), nullable=True)
    completed_at = Column(String(64), nullable=True)
    error = Column(Text, nullable=True)
    worker_id = Column(String(255), nullable=True)


class Job(Base):
    """Mirror of the Genlio Job model to get dataset paths."""
    __tablename__ = "jobs"

    job_id = Column(String(255), primary_key=True)
    dataset_id = Column(String(255), nullable=True)
    version = Column(String(255), nullable=True)
    status = Column(String(32), nullable=False)
    task_type = Column(String(64), nullable=False)
    result_path = Column(Text, nullable=True)
    extra_data = Column(JSON, nullable=True)
    created_at = Column(String(64), nullable=False)


class Dataset(Base):
    """Mirror of the Genlio Dataset model."""
    __tablename__ = "datasets"

    dataset_id = Column(String(255), primary_key=True)
    latest_version = Column(String(255), nullable=True)


class WorkerStatus(Base):
    """Worker status for monitoring."""
    __tablename__ = "worker_status"

    worker_id = Column(String(255), primary_key=True)
    worker_type = Column(String(32), nullable=False)
    worker_name = Column(String(255), nullable=True)
    status = Column(String(32), nullable=False, default="offline")
    current_job_id = Column(String(255), nullable=True)
    current_job_name = Column(String(255), nullable=True)
    current_job_type = Column(String(32), nullable=True)  # "training" or "inference"
    hostname = Column(String(255), nullable=True)
    ip_address = Column(String(64), nullable=True)
    cpu_count = Column(Integer, nullable=True)
    cpu_percent = Column(Float, nullable=True)
    memory_total_gb = Column(Float, nullable=True)
    memory_used_gb = Column(Float, nullable=True)
    memory_percent = Column(Float, nullable=True)
    gpu_count = Column(Integer, nullable=True)
    gpus = Column(JSON, nullable=True)
    disk_total_gb = Column(Float, nullable=True)
    disk_used_gb = Column(Float, nullable=True)
    disk_percent = Column(Float, nullable=True)
    python_version = Column(String(32), nullable=True)
    torch_version = Column(String(32), nullable=True)
    cuda_version = Column(String(32), nullable=True)
    jobs_completed = Column(Integer, nullable=True, default=0)
    jobs_failed = Column(Integer, nullable=True, default=0)
    training_jobs_completed = Column(Integer, nullable=True, default=0)
    inference_jobs_completed = Column(Integer, nullable=True, default=0)
    started_at = Column(String(64), nullable=False)
    last_heartbeat = Column(String(64), nullable=False)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class JobType(Enum):
    TRAINING = "training"
    INFERENCE = "inference"


@dataclass
class JobSpec:
    """Specification for a claimed job."""
    job_type: JobType
    job_id: str
    job_name: Optional[str]


@dataclass
class WorkerConfig:
    """Worker configuration."""
    database_url: str
    worker_id: str
    worker_name: Optional[str]
    poll_interval: int
    data_dir: Path
    output_dir: Path
    supabase_url: str
    supabase_key: str
    model_cache_dir: Path
    training_min_gpu_gb: float
    inference_min_gpu_gb: float
    inference_in_subprocess: bool


# =============================================================================
# Utility Functions
# =============================================================================

def _now_iso() -> str:
    """Get current time in ISO format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _get_ip_address() -> str:
    """Get the local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _collect_static_info() -> Dict[str, Any]:
    """Collect static system information (doesn't change during runtime)."""
    info = {
        "hostname": socket.gethostname(),
        "ip_address": _get_ip_address(),
        "cpu_count": psutil.cpu_count(logical=True),
        "python_version": platform.python_version(),
    }
    
    # Get memory total
    mem = psutil.virtual_memory()
    info["memory_total_gb"] = round(mem.total / (1024**3), 2)
    
    # Get disk total
    disk = psutil.disk_usage("/")
    info["disk_total_gb"] = round(disk.total / (1024**3), 2)
    
    # Try to get torch/cuda versions
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    
    return info


def _collect_gpu_metrics() -> Tuple[int, List[Dict[str, Any]]]:
    """Collect comprehensive GPU metrics using pynvml (nvidia-smi) and torch."""
    gpus = []
    gpu_count = 0
    
    # First try pynvml for accurate system-wide GPU metrics
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get GPU name
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_total_gb = mem_info.total / (1024**3)
            mem_used_gb = mem_info.used / (1024**3)
            mem_free_gb = (mem_info.total - mem_info.used) / (1024**3)
            mem_percent = (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0
            
            gpu_info = {
                "index": i,
                "name": name,
                "memory_total_gb": round(mem_total_gb, 2),
                "memory_used_gb": round(mem_used_gb, 2),
                "memory_free_gb": round(mem_free_gb, 2),
                "memory_percent": round(mem_percent, 1),
            }
            
            # Get GPU utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_info["utilization_gpu"] = util.gpu
                gpu_info["utilization_memory"] = util.memory
            except Exception:
                pass
            
            # Get temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_info["temperature_c"] = temp
            except Exception:
                pass
            
            # Get power usage
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                gpu_info["power_watts"] = round(power / 1000, 1)
                gpu_info["power_limit_watts"] = round(power_limit / 1000, 1)
            except Exception:
                pass
            
            # Get fan speed
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                gpu_info["fan_speed_percent"] = fan_speed
            except Exception:
                pass
            
            # Get compute processes
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                gpu_info["process_count"] = len(processes)
            except Exception:
                pass
            
            # Get PyTorch-specific memory if available
            try:
                import torch
                if torch.cuda.is_available() and i < torch.cuda.device_count():
                    gpu_info["pytorch_allocated_gb"] = round(torch.cuda.memory_allocated(i) / (1024**3), 2)
                    gpu_info["pytorch_reserved_gb"] = round(torch.cuda.memory_reserved(i) / (1024**3), 2)
            except Exception:
                pass
            
            gpus.append(gpu_info)
            
    except ImportError:
        # Fall back to torch-only metrics
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    mem_total = props.total_memory / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    
                    gpu_info = {
                        "index": i,
                        "name": props.name,
                        "memory_total_gb": round(mem_total, 2),
                        "memory_used_gb": round(mem_reserved, 2),
                        "memory_free_gb": round(mem_total - mem_reserved, 2),
                        "memory_percent": round((mem_reserved / mem_total) * 100, 1) if mem_total > 0 else 0,
                        "pytorch_allocated_gb": round(torch.cuda.memory_allocated(i) / (1024**3), 2),
                        "pytorch_reserved_gb": round(mem_reserved, 2),
                    }
                    gpus.append(gpu_info)
        except ImportError:
            pass
    except Exception as e:
        logger.warning(f"Error collecting GPU metrics: {e}")
    
    return gpu_count, gpus


def _collect_dynamic_metrics() -> Dict[str, Any]:
    """Collect dynamic system metrics."""
    metrics = {}
    
    # CPU usage
    metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
    
    # Memory usage
    mem = psutil.virtual_memory()
    metrics["memory_used_gb"] = round(mem.used / (1024**3), 2)
    metrics["memory_percent"] = mem.percent
    
    # Disk usage
    disk = psutil.disk_usage("/")
    metrics["disk_used_gb"] = round(disk.used / (1024**3), 2)
    metrics["disk_percent"] = round(disk.percent, 1)
    
    # GPU metrics
    gpu_count, gpus = _collect_gpu_metrics()
    metrics["gpu_count"] = gpu_count
    metrics["gpus"] = gpus
    
    return metrics


# =============================================================================
# Status Reporter
# =============================================================================

class StatusReporter:
    """Reports worker status and system metrics to the database."""

    def __init__(
        self,
        session_factory,
        worker_id: str,
        worker_type: str,
        worker_name: Optional[str] = None,
        heartbeat_interval: float = 5.0,
    ):
        self.session_factory = session_factory
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.worker_name = worker_name or f"{worker_type}-{worker_id[:8]}"
        self.heartbeat_interval = heartbeat_interval
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_job_id: Optional[str] = None
        self._current_job_name: Optional[str] = None
        self._current_job_type: Optional[str] = None
        self._jobs_completed = 0
        self._jobs_failed = 0
        self._training_jobs_completed = 0
        self._inference_jobs_completed = 0
        self._started_at = _now_iso()
        
        # Cache static info
        self._static_info = _collect_static_info()

    def _update_status(self):
        """Update worker status in the database."""
        try:
            dynamic_metrics = _collect_dynamic_metrics()
            
            with self.session_factory() as session:
                worker_status = session.get(WorkerStatus, self.worker_id)
                
                if worker_status is None:
                    worker_status = WorkerStatus(
                        worker_id=self.worker_id,
                        worker_type=self.worker_type,
                        worker_name=self.worker_name,
                        started_at=self._started_at,
                    )
                    session.add(worker_status)
                
                # Update static info
                worker_status.hostname = self._static_info.get("hostname")
                worker_status.ip_address = self._static_info.get("ip_address")
                worker_status.cpu_count = self._static_info.get("cpu_count")
                worker_status.memory_total_gb = self._static_info.get("memory_total_gb")
                worker_status.disk_total_gb = self._static_info.get("disk_total_gb")
                worker_status.python_version = self._static_info.get("python_version")
                worker_status.torch_version = self._static_info.get("torch_version")
                worker_status.cuda_version = self._static_info.get("cuda_version")
                
                # Update dynamic metrics
                worker_status.cpu_percent = dynamic_metrics.get("cpu_percent")
                worker_status.memory_used_gb = dynamic_metrics.get("memory_used_gb")
                worker_status.memory_percent = dynamic_metrics.get("memory_percent")
                worker_status.disk_used_gb = dynamic_metrics.get("disk_used_gb")
                worker_status.disk_percent = dynamic_metrics.get("disk_percent")
                worker_status.gpu_count = dynamic_metrics.get("gpu_count")
                worker_status.gpus = dynamic_metrics.get("gpus")
                
                # Update job info
                worker_status.status = "busy" if self._current_job_id else "online"
                worker_status.current_job_id = self._current_job_id
                worker_status.current_job_name = self._current_job_name
                worker_status.current_job_type = self._current_job_type
                worker_status.jobs_completed = self._jobs_completed
                worker_status.jobs_failed = self._jobs_failed
                worker_status.training_jobs_completed = self._training_jobs_completed
                worker_status.inference_jobs_completed = self._inference_jobs_completed
                
                # Update heartbeat
                worker_status.last_heartbeat = _now_iso()
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Failed to update worker status: {e}")

    def _heartbeat_loop(self):
        """Background thread that sends periodic heartbeats."""
        while self._running:
            self._update_status()
            time.sleep(self.heartbeat_interval)

    def start(self):
        """Start the status reporter background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        logger.info(f"Status reporter started for worker {self.worker_id}")

    def stop(self):
        """Stop the status reporter and mark worker as offline."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        
        # Mark as offline
        try:
            with self.session_factory() as session:
                worker_status = session.get(WorkerStatus, self.worker_id)
                if worker_status:
                    worker_status.status = "offline"
                    worker_status.current_job_id = None
                    worker_status.current_job_name = None
                    worker_status.current_job_type = None
                    worker_status.last_heartbeat = _now_iso()
                    session.commit()
        except Exception as e:
            logger.error(f"Failed to mark worker as offline: {e}")
        
        logger.info(f"Status reporter stopped for worker {self.worker_id}")

    def set_current_job(
        self, 
        job_id: Optional[str], 
        job_name: Optional[str] = None,
        job_type: Optional[str] = None,
    ):
        """Set the current job being processed."""
        self._current_job_id = job_id
        self._current_job_name = job_name
        self._current_job_type = job_type

    def record_job_completed(self, success: bool = True, job_type: Optional[JobType] = None):
        """Record a job completion."""
        if success:
            self._jobs_completed += 1
            if job_type == JobType.TRAINING:
                self._training_jobs_completed += 1
            elif job_type == JobType.INFERENCE:
                self._inference_jobs_completed += 1
        else:
            self._jobs_failed += 1


# =============================================================================
# Training Progress Callback
# =============================================================================

class GenlioProgressCallback(TrainerCallback):
    """Custom TrainerCallback that reports training progress to the database."""
    
    def __init__(self, session_factory, job_id: str):
        super().__init__()
        self.session_factory = session_factory
        self.job_id = job_id
        self.training_started = False
        
        self.train_history: list[dict] = []
        self.eval_history: list[dict] = []
        
        self.last_db_update_time = 0.0
        self.db_update_interval = 3.0
        
        logger.info(f"[GenlioProgressCallback] Created for job_id={job_id}")
    
    def _update_job_status(self, **kwargs):
        """Update job status in database."""
        try:
            with self.session_factory() as session:
                job = session.get(FineTuningJob, self.job_id)
                if not job:
                    return
                
                for key, value in kwargs.items():
                    if value is not None and hasattr(job, key):
                        setattr(job, key, value)
                
                job.updated_at = _now_iso()
                session.commit()
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")
    
    def _build_metrics_payload(self, current_metrics: dict) -> dict:
        return {
            "current": current_metrics,
            "history": self.train_history,
            "eval_history": self.eval_history,
        }
    
    def _should_update_db(self) -> bool:
        current_time = time.time()
        if current_time - self.last_db_update_time >= self.db_update_interval:
            self.last_db_update_time = current_time
            return True
        return False
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.training_started = True
        self.last_db_update_time = time.time()
        
        logger.info(f"[Callback] Training started: max_steps={state.max_steps}")
        
        initial_metrics = self._build_metrics_payload({
            "loss": None,
            "learning_rate": None,
            "epoch": 0.0,
        })
        
        self._update_job_status(
            current_step=0,
            total_steps=state.max_steps,
            current_epoch=0.0,
            total_epochs=float(args.num_train_epochs),
            metrics=initial_metrics,
        )
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        
        current_step = getattr(state, "global_step", 0)
        max_steps = getattr(state, "max_steps", 0)
        timestamp = _now_iso()
        
        loss = logs.get("loss")
        learning_rate = logs.get("learning_rate")
        epoch = logs.get("epoch")
        grad_norm = logs.get("grad_norm")
        eval_loss = logs.get("eval_loss")
        
        if loss is not None:
            train_record = {"step": current_step, "timestamp": timestamp}
            if loss is not None:
                train_record["loss"] = float(loss)
            if learning_rate is not None:
                train_record["learning_rate"] = float(learning_rate)
            if epoch is not None:
                train_record["epoch"] = float(epoch)
            if grad_norm is not None:
                train_record["grad_norm"] = float(grad_norm)
            
            self.train_history.append(train_record)
        
        if eval_loss is not None:
            eval_record = {"step": current_step, "timestamp": timestamp, "eval_loss": float(eval_loss)}
            for key, value in logs.items():
                if key.startswith("eval_") and key != "eval_loss" and isinstance(value, (int, float)):
                    eval_record[key] = float(value)
            self.eval_history.append(eval_record)
        
        current_metrics = {}
        if loss is not None:
            current_metrics["loss"] = float(loss)
        if learning_rate is not None:
            current_metrics["learning_rate"] = float(learning_rate)
        if epoch is not None:
            current_metrics["epoch"] = float(epoch)
        if grad_norm is not None:
            current_metrics["grad_norm"] = float(grad_norm)
        if eval_loss is not None:
            current_metrics["eval_loss"] = float(eval_loss)
        
        if self._should_update_db():
            metrics_payload = self._build_metrics_payload(current_metrics)
            self._update_job_status(
                current_step=current_step,
                total_steps=max_steps,
                current_epoch=float(epoch) if epoch is not None else None,
                metrics=metrics_payload,
            )
    
    def on_train_end(self, args, state, control, **kwargs):
        if not self.training_started:
            return
        
        logger.info(f"[Callback] Training ended: final_step={state.global_step}")
        
        current_metrics = {}
        if state.log_history:
            last_log = state.log_history[-1]
            if "loss" in last_log:
                current_metrics["loss"] = float(last_log["loss"])
            if "learning_rate" in last_log:
                current_metrics["learning_rate"] = float(last_log["learning_rate"])
            if "epoch" in last_log:
                current_metrics["epoch"] = float(last_log["epoch"])
        
        metrics_payload = self._build_metrics_payload(current_metrics)
        self._update_job_status(
            current_step=state.global_step,
            total_steps=state.max_steps,
            current_epoch=float(state.epoch) if hasattr(state, "epoch") and state.epoch else None,
            metrics=metrics_payload,
        )
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        
        current_step = getattr(state, "global_step", 0)
        timestamp = _now_iso()
        
        eval_record = {"step": current_step, "timestamp": timestamp}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                eval_record[key] = float(value)
        
        if len(eval_record) > 2:
            self.eval_history.append(eval_record)
            self.last_db_update_time = 0
            
            current_metrics = {"eval_step": current_step}
            current_metrics.update({k: v for k, v in eval_record.items() if k not in ["step", "timestamp"]})
            
            metrics_payload = self._build_metrics_payload(current_metrics)
            self._update_job_status(metrics=metrics_payload)


# =============================================================================
# Model Cache (for inference)
# =============================================================================

class ModelCache:
    """Cache for loaded models to avoid reloading for consecutive inference requests."""
    
    def __init__(self):
        self.current_model_key: Optional[str] = None
        self.chat_model: Optional[Any] = None
    
    def get_model_key(self, base_model: str, adapter_path: Optional[str]) -> str:
        return f"{base_model}:{adapter_path or 'none'}"
    
    def get_or_load(
        self, 
        base_model: str, 
        adapter_path: Optional[str],
        template: str,
        local_adapter_path: Optional[str] = None,
    ) -> Any:
        """Get cached model or load a new one."""
        from llamafactory.chat import ChatModel
        from llamafactory.extras.misc import torch_gc
        
        model_key = self.get_model_key(base_model, adapter_path)
        
        if self.current_model_key == model_key and self.chat_model is not None:
            logger.info(f"Using cached model: {model_key}")
            return self.chat_model
        
        if self.chat_model is not None:
            logger.info(f"Unloading previous model: {self.current_model_key}")
            del self.chat_model
            self.chat_model = None
            torch_gc()
        
        args = {
            "model_name_or_path": base_model,
            "template": template,
            "finetuning_type": "lora" if local_adapter_path else "full",
            "infer_backend": "huggingface",
        }
        
        if local_adapter_path:
            args["adapter_name_or_path"] = local_adapter_path
        
        logger.info(f"Loading model: {base_model} with adapter: {adapter_path}")
        self.chat_model = ChatModel(args)
        self.current_model_key = model_key
        
        return self.chat_model
    
    def clear(self):
        """Clear the model cache."""
        if self.chat_model is not None:
            from llamafactory.extras.misc import torch_gc
            del self.chat_model
            self.chat_model = None
            self.current_model_key = None
            torch_gc()


# =============================================================================
# Worker
# =============================================================================

class GenlioWorker:
    """Single worker that handles both training and inference jobs."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.engine = create_engine(config.database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self.running = True
        self.current_job_spec: Optional[JobSpec] = None
        
        # HTTP client for Supabase
        self.http_client = httpx.Client(timeout=300)
        
        # Model cache for inference (kept in main process)
        self.model_cache = ModelCache()
        
        # Single status reporter
        self.status_reporter = StatusReporter(
            session_factory=self.SessionLocal,
            worker_id=config.worker_id,
            worker_type="worker",
            worker_name=config.worker_name,
            heartbeat_interval=5.0,
        )
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Ensure directories exist
        config.data_dir.mkdir(parents=True, exist_ok=True)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Worker {config.worker_id} initialized")
        logger.info(f"Worker name: {config.worker_name or 'auto'}")
        logger.info(f"Data directory: {config.data_dir}")
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Model cache directory: {config.model_cache_dir}")
        logger.info(f"Training min GPU: {config.training_min_gpu_gb}GB")
        logger.info(f"Inference min GPU: {config.inference_min_gpu_gb}GB")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
        self.status_reporter.stop()
        
        if self.current_job_spec:
            self._mark_job_failed(
                self.current_job_spec,
                "Worker interrupted by signal"
            )

    def _check_resources_available(self, job_type: JobType) -> Tuple[bool, str]:
        """Check if sufficient GPU resources exist for the job type."""
        gpu_count, gpus = _collect_gpu_metrics()
        
        if not gpus:
            return False, "No GPUs available"
        
        min_required = (
            self.config.training_min_gpu_gb 
            if job_type == JobType.TRAINING 
            else self.config.inference_min_gpu_gb
        )
        
        for gpu in gpus:
            mem_free = gpu.get("memory_free_gb", 0)
            if mem_free >= min_required:
                return True, f"GPU {gpu['index']} has {mem_free:.1f}GB free (need {min_required}GB)"
        
        best_available = max(g.get("memory_free_gb", 0) for g in gpus)
        return False, f"Insufficient GPU memory (need {min_required}GB, best available: {best_available:.1f}GB)"

    def _claim_next_job(self) -> Optional[JobSpec]:
        """Claim next pending job. Priority: training > inference."""
        with self.SessionLocal() as session:
            # First try training jobs
            training_job = (
                session.query(FineTuningJob)
                .filter(FineTuningJob.status == "pending")
                .order_by(FineTuningJob.created_at.asc())
                .first()
            )
            
            if training_job:
                can_run, reason = self._check_resources_available(JobType.TRAINING)
                if can_run:
                    training_job.status = "queued"
                    training_job.worker_id = self.config.worker_id
                    training_job.updated_at = _now_iso()
                    session.commit()
                    logger.info(f"Claimed training job {training_job.job_id}: {reason}")
                    return JobSpec(
                        job_type=JobType.TRAINING,
                        job_id=training_job.job_id,
                        job_name=training_job.job_name,
                    )
                else:
                    logger.debug(f"Cannot run training job: {reason}")
            
            # Then try inference jobs
            inference_job = (
                session.query(InferenceJob)
                .filter(InferenceJob.status == "pending")
                .order_by(InferenceJob.created_at.asc())
                .first()
            )
            
            if inference_job:
                can_run, reason = self._check_resources_available(JobType.INFERENCE)
                if can_run:
                    inference_job.status = "queued"
                    inference_job.worker_id = self.config.worker_id
                    inference_job.updated_at = _now_iso()
                    session.commit()
                    logger.info(f"Claimed inference job {inference_job.job_id}: {reason}")
                    return JobSpec(
                        job_type=JobType.INFERENCE,
                        job_id=inference_job.job_id,
                        job_name=inference_job.job_name,
                    )
                else:
                    logger.debug(f"Cannot run inference job: {reason}")
            
            return None

    def _mark_job_failed(self, job_spec: JobSpec, error: str):
        """Mark a job as failed."""
        with self.SessionLocal() as session:
            if job_spec.job_type == JobType.TRAINING:
                job = session.get(FineTuningJob, job_spec.job_id)
            else:
                job = session.get(InferenceJob, job_spec.job_id)
            
            if job:
                job.status = "failed"
                job.error = error
                job.completed_at = _now_iso()
                job.updated_at = _now_iso()
                session.commit()

    # -------------------------------------------------------------------------
    # Training Job Processing
    # -------------------------------------------------------------------------

    def _download_from_supabase(self, storage_path: str) -> bytes:
        """Download a file from Supabase storage."""
        url = f"{self.config.supabase_url}/storage/v1/object/{STORAGE_BUCKET}/{storage_path}"
        headers = {
            "Authorization": f"Bearer {self.config.supabase_key}",
            "apikey": self.config.supabase_key,
        }
        response = self.http_client.get(url, headers=headers)
        response.raise_for_status()
        return response.content

    def _upload_to_supabase(self, bucket: str, storage_path: str, file_content: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload a file to Supabase storage."""
        url = f"{self.config.supabase_url}/storage/v1/object/{bucket}/{storage_path}"
        headers = {
            "Authorization": f"Bearer {self.config.supabase_key}",
            "apikey": self.config.supabase_key,
            "Content-Type": content_type,
        }
        response = self.http_client.post(url, headers=headers, content=file_content)
        if response.status_code == 400:
            response = self.http_client.put(url, headers=headers, content=file_content)
        response.raise_for_status()
        return storage_path

    def _get_dataset_storage_path(self, job: FineTuningJob) -> Optional[str]:
        """Get the storage path for a dataset from the jobs table."""
        with self.SessionLocal() as session:
            version = job.version
            if not version:
                dataset = session.get(Dataset, job.dataset_id)
                if dataset:
                    version = dataset.latest_version
            
            if not version:
                logger.warning(f"Could not resolve version for dataset {job.dataset_id}")
                return None
            
            source_job = (
                session.query(Job)
                .filter(
                    Job.dataset_id == job.dataset_id,
                    Job.version == version,
                    Job.status == "succeeded",
                    Job.task_type.in_(["upload_dataset", "generate_from_summaries", "deidentify_documents"]),
                )
                .order_by(Job.created_at.desc())
                .first()
            )
            
            if not source_job:
                logger.warning(f"No source job found for dataset {job.dataset_id}/{version}")
                return None
            
            storage_path = source_job.result_path
            if not storage_path and source_job.extra_data:
                storage_path = source_job.extra_data.get("data_path")
            
            logger.info(f"Found dataset storage path: {storage_path}")
            return storage_path

    def _prepare_dataset(self, job: FineTuningJob) -> str:
        """Download and prepare dataset for training."""
        dataset_name = f"genlio_{job.job_id}"
        dataset_dir = self.config.data_dir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_file = dataset_dir / f"{dataset_name}.json"
        storage_path = self._get_dataset_storage_path(job)
        
        if storage_path:
            try:
                logger.info(f"Downloading dataset from Supabase: {storage_path}")
                csv_bytes = self._download_from_supabase(storage_path)
                csv_content = csv_bytes.decode("utf-8")
                
                original_limit = csv.field_size_limit()
                try:
                    csv.field_size_limit(sys.maxsize)
                    reader = csv.DictReader(io.StringIO(csv_content))
                    
                    dataset_config = job.dataset_config or {}
                    prompt_col = dataset_config.get("prompt_column", "instruction")
                    response_col = dataset_config.get("response_column", "output")
                    input_col = dataset_config.get("input_column", "input")
                    system_col = dataset_config.get("system_column")
                    
                    records = []
                    for row in reader:
                        record = {
                            "instruction": row.get(prompt_col, ""),
                            "input": row.get(input_col, ""),
                            "output": row.get(response_col, ""),
                        }
                        if system_col and system_col in row:
                            record["system"] = row.get(system_col, "")
                        records.append(record)
                    
                    with open(dataset_file, "w", encoding="utf-8") as f:
                        json.dump(records, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Downloaded and converted dataset with {len(records)} records")
                finally:
                    csv.field_size_limit(original_limit)
                    
            except Exception as e:
                logger.exception(f"Failed to download dataset from Supabase: {e}")
                raise RuntimeError(f"Failed to download dataset: {e}")
        else:
            raise RuntimeError(f"Could not find storage path for dataset {job.dataset_id}")
        
        # Update dataset_info.json
        dataset_info_path = dataset_dir / "dataset_info.json"
        if dataset_info_path.exists():
            with open(dataset_info_path) as f:
                dataset_info = json.load(f)
        else:
            dataset_info = {}
        
        dataset_info[dataset_name] = {"file_name": f"{dataset_name}.json"}
        
        with open(dataset_info_path, "w") as f:
            json.dump(dataset_info, f, indent=2)
        
        return dataset_name

    def _build_training_args(self, job: FineTuningJob, dataset_name: str) -> Dict[str, Any]:
        """Build LLaMA-Factory training arguments."""
        training_config = job.training_config or {}
        lora_config = job.lora_config or {}
        dataset_config = job.dataset_config or {}
        
        output_dir = self.config.output_dir / job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        do_eval = training_config.get("do_eval", True)
        eval_strategy = training_config.get("eval_strategy", "steps")
        eval_steps = training_config.get("eval_steps", 100)
        val_size = training_config.get("val_size", 0.1)
        
        args = {
            "model_name_or_path": job.base_model,
            "trust_remote_code": True,
            "stage": job.stage,
            "do_train": True,
            "finetuning_type": job.finetuning_type,
            "dataset": dataset_name,
            "dataset_dir": str(self.config.data_dir),
            "template": dataset_config.get("template", "llama3"),
            "cutoff_len": dataset_config.get("cutoff_len", 2048),
            "overwrite_cache": True,
            "preprocessing_num_workers": 4,
            "output_dir": str(output_dir),
            "logging_steps": training_config.get("logging_steps", 10),
            "save_steps": training_config.get("save_steps", 500),
            "plot_loss": True,
            "overwrite_output_dir": True,
            "per_device_train_batch_size": training_config.get("per_device_train_batch_size", 1),
            "gradient_accumulation_steps": training_config.get("gradient_accumulation_steps", 8),
            "learning_rate": training_config.get("learning_rate", 1e-4),
            "num_train_epochs": training_config.get("num_train_epochs", 3.0),
            "lr_scheduler_type": training_config.get("lr_scheduler_type", "cosine"),
            "warmup_ratio": training_config.get("warmup_ratio", 0.1),
            "max_grad_norm": training_config.get("max_grad_norm", 1.0),
            "bf16": training_config.get("bf16", True),
            "fp16": training_config.get("fp16", False),
            "report_to": "none",
        }
        
        if do_eval:
            args["do_eval"] = True
            args["eval_strategy"] = eval_strategy
            args["eval_steps"] = eval_steps
            args["val_size"] = val_size
            args["per_device_eval_batch_size"] = training_config.get("per_device_eval_batch_size", 1)
        
        if job.finetuning_type == "lora":
            args["lora_rank"] = lora_config.get("lora_rank", 8)
            lora_alpha = lora_config.get("lora_alpha")
            if lora_alpha:
                args["lora_alpha"] = lora_alpha
            args["lora_dropout"] = lora_config.get("lora_dropout", 0.0)
            args["lora_target"] = lora_config.get("lora_target", "all")
        
        return args

    def _upload_adapter_to_storage(self, job: FineTuningJob, local_adapter_path: str) -> str:
        """Upload adapter files to Supabase storage."""
        adapter_dir = Path(local_adapter_path)
        storage_prefix = f"{job.dataset_id}/{job.job_id}"
        
        essential_patterns = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "training_args.bin",
        ]
        
        uploaded_files = []
        
        for file_path in adapter_dir.iterdir():
            if not file_path.is_file():
                continue
            
            should_upload = (
                file_path.name in essential_patterns or
                file_path.suffix in [".json", ".safetensors", ".bin"]
            )
            
            if should_upload:
                storage_path = f"{storage_prefix}/{file_path.name}"
                content_type = "application/json" if file_path.suffix == ".json" else "application/octet-stream"
                
                logger.info(f"Uploading {file_path.name} to storage...")
                
                with open(file_path, "rb") as f:
                    file_content = f.read()
                
                self._upload_to_supabase(ADAPTERS_BUCKET, storage_path, file_content, content_type)
                uploaded_files.append(file_path.name)
        
        logger.info(f"Uploaded {len(uploaded_files)} adapter files to {storage_prefix}")
        return storage_prefix

    def _process_training_job(self, job_id: str):
        """Process a training job."""
        with self.SessionLocal() as session:
            job = session.get(FineTuningJob, job_id)
            if not job:
                logger.error(f"Training job {job_id} not found")
                return
            
            # Detach from session for processing
            session.expunge(job)
        
        logger.info(f"Processing training job {job.job_id}: {job.job_name or 'Unnamed'}")
        
        try:
            # Update status to running
            with self.SessionLocal() as session:
                db_job = session.get(FineTuningJob, job.job_id)
                db_job.status = "running"
                db_job.started_at = _now_iso()
                db_job.updated_at = _now_iso()
                session.commit()
            
            # Import LLaMA-Factory
            from llamafactory.train.tuner import run_exp
            
            # Prepare dataset
            dataset_name = self._prepare_dataset(job)
            
            # Build training arguments
            args = self._build_training_args(job, dataset_name)
            output_dir = args["output_dir"]
            
            # Update job with output directory
            with self.SessionLocal() as session:
                db_job = session.get(FineTuningJob, job.job_id)
                db_job.output_dir = output_dir
                db_job.total_epochs = args["num_train_epochs"]
                db_job.updated_at = _now_iso()
                session.commit()
            
            logger.info(f"Starting training with args: {json.dumps(args, indent=2)}")
            
            # Create progress callback
            progress_callback = GenlioProgressCallback(
                session_factory=self.SessionLocal,
                job_id=job.job_id,
            )
            
            # Run training
            run_exp(args=args, callbacks=[progress_callback])
            
            # Upload adapters
            adapter_path = output_dir
            try:
                logger.info("Uploading adapters to storage...")
                storage_path = self._upload_adapter_to_storage(job, output_dir)
                adapter_path = storage_path
                logger.info(f"Adapters uploaded to: {storage_path}")
            except Exception as upload_error:
                logger.exception(f"Failed to upload adapters: {upload_error}")
                logger.warning(f"Using local adapter path: {adapter_path}")
            
            # Mark as succeeded
            with self.SessionLocal() as session:
                db_job = session.get(FineTuningJob, job.job_id)
                db_job.status = "succeeded"
                db_job.adapter_path = adapter_path
                db_job.completed_at = _now_iso()
                db_job.updated_at = _now_iso()
                session.commit()
            
            logger.info(f"Training job {job.job_id} completed successfully")
            
        except Exception as e:
            logger.exception(f"Error processing training job {job.job_id}: {e}")
            with self.SessionLocal() as session:
                db_job = session.get(FineTuningJob, job.job_id)
                db_job.status = "failed"
                db_job.error = str(e)
                db_job.completed_at = _now_iso()
                db_job.updated_at = _now_iso()
                session.commit()
            raise

    # -------------------------------------------------------------------------
    # Inference Job Processing
    # -------------------------------------------------------------------------

    def _download_adapter(self, storage_path: str) -> Path:
        """Download adapter files from Supabase storage."""
        local_dir = self.config.model_cache_dir / storage_path.replace("/", "_")
        
        adapter_config = local_dir / "adapter_config.json"
        if adapter_config.exists():
            logger.info(f"Using cached adapter: {local_dir}")
            return local_dir
        
        local_dir.mkdir(parents=True, exist_ok=True)
        
        essential_files = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
        ]
        
        for filename in essential_files:
            try:
                url = f"{self.config.supabase_url}/storage/v1/object/{ADAPTERS_BUCKET}/{storage_path}/{filename}"
                headers = {
                    "Authorization": f"Bearer {self.config.supabase_key}",
                    "apikey": self.config.supabase_key,
                }
                
                response = self.http_client.get(url, headers=headers)
                if response.status_code == 200:
                    local_path = local_dir / filename
                    with open(local_path, "wb") as f:
                        f.write(response.content)
                    logger.info(f"Downloaded {filename}")
            except Exception as e:
                logger.warning(f"Could not download {filename}: {e}")
        
        return local_dir

    def _run_single_inference(self, job: InferenceJob, chat_model: Any) -> Dict[str, Any]:
        """Run single-prompt inference."""
        messages = job.messages or []
        system = job.system_prompt
        gen_config = job.generation_config or {}
        
        chat_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]
        
        start_time = time.time()
        responses = chat_model.chat(
            chat_messages,
            system=system,
            max_new_tokens=gen_config.get("max_new_tokens", 512),
            temperature=gen_config.get("temperature", 0.7),
            top_p=gen_config.get("top_p", 0.9),
            top_k=gen_config.get("top_k", 50),
            do_sample=gen_config.get("do_sample", True),
            repetition_penalty=gen_config.get("repetition_penalty", 1.0),
        )
        
        latency = time.time() - start_time
        response = responses[0] if responses else None
        
        return {
            "response": response.response_text if response else "",
            "prompt_tokens": response.prompt_length if response else 0,
            "response_tokens": response.response_length if response else 0,
            "latency_seconds": round(latency, 3),
            "finish_reason": response.finish_reason if response else "unknown",
        }

    def _run_batch_inference(self, job: InferenceJob, chat_model: Any, session_factory) -> Dict[str, Any]:
        """Run batch inference on prompts."""
        results = []
        gen_config = job.generation_config or {}
        system = job.system_prompt
        total_latency = 0
        
        prompts = job.messages or []
        
        with session_factory() as session:
            db_job = session.get(InferenceJob, job.job_id)
            db_job.rows_total = len(prompts)
            db_job.updated_at = _now_iso()
            session.commit()
        
        for i, prompt_data in enumerate(prompts):
            try:
                prompt = prompt_data.get("content", "") if isinstance(prompt_data, dict) else str(prompt_data)
                
                start_time = time.time()
                responses = chat_model.chat(
                    [{"role": "user", "content": prompt}],
                    system=system,
                    max_new_tokens=gen_config.get("max_new_tokens", 512),
                    temperature=gen_config.get("temperature", 0.7),
                    do_sample=gen_config.get("do_sample", True),
                )
                latency = time.time() - start_time
                total_latency += latency
                
                response = responses[0] if responses else None
                
                result = {
                    "index": i,
                    "prompt": prompt,
                    "response": response.response_text if response else "",
                    "latency_seconds": round(latency, 3),
                }
                
                if job.expected_column and isinstance(prompt_data, dict):
                    result["expected"] = prompt_data.get("expected", "")
                
                results.append(result)
                
                with session_factory() as session:
                    db_job = session.get(InferenceJob, job.job_id)
                    db_job.rows_processed = i + 1
                    db_job.updated_at = _now_iso()
                    session.commit()
                
            except Exception as e:
                logger.exception(f"Error processing prompt {i}: {e}")
                results.append({
                    "index": i,
                    "prompt": prompt_data,
                    "error": str(e),
                })
        
        return {
            "results": results,
            "metrics": {
                "total_prompts": len(prompts),
                "successful": len([r for r in results if "error" not in r]),
                "avg_latency_seconds": round(total_latency / len(prompts), 3) if prompts else 0,
            },
        }

    def _process_inference_job(self, job_id: str):
        """Process an inference job (in main process with model cache)."""
        with self.SessionLocal() as session:
            job = session.get(InferenceJob, job_id)
            if not job:
                logger.error(f"Inference job {job_id} not found")
                return
            
            session.expunge(job)
        
        logger.info(f"Processing inference job {job.job_id}: {job.job_name or 'Unnamed'}")
        
        try:
            # Update status to running
            with self.SessionLocal() as session:
                db_job = session.get(InferenceJob, job.job_id)
                db_job.status = "running"
                db_job.started_at = _now_iso()
                db_job.updated_at = _now_iso()
                session.commit()
            
            # Download adapter if needed
            local_adapter_path = None
            if job.adapter_path:
                local_adapter_path = str(self._download_adapter(job.adapter_path))
            
            # Load or get cached model
            chat_model = self.model_cache.get_or_load(
                base_model=job.base_model,
                adapter_path=job.adapter_path,
                template=job.template or "llama3",
                local_adapter_path=local_adapter_path,
            )
            
            # Run inference
            if job.inference_mode == "single":
                output = self._run_single_inference(job, chat_model)
            else:
                output = self._run_batch_inference(job, chat_model, self.SessionLocal)
            
            # Mark as succeeded
            with self.SessionLocal() as session:
                db_job = session.get(InferenceJob, job.job_id)
                db_job.status = "succeeded"
                db_job.output = output
                db_job.completed_at = _now_iso()
                db_job.updated_at = _now_iso()
                session.commit()
            
            logger.info(f"Inference job {job.job_id} completed successfully")
            
        except Exception as e:
            logger.exception(f"Error processing inference job {job.job_id}: {e}")
            with self.SessionLocal() as session:
                db_job = session.get(InferenceJob, job.job_id)
                db_job.status = "failed"
                db_job.error = str(e)
                db_job.completed_at = _now_iso()
                db_job.updated_at = _now_iso()
                session.commit()
            raise

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------

    def _process_job(self, job_spec: JobSpec):
        """Process a job based on its type."""
        self.current_job_spec = job_spec
        self.status_reporter.set_current_job(
            job_spec.job_id,
            job_spec.job_name,
            job_type=job_spec.job_type.value,
        )
        
        job_success = False
        try:
            if job_spec.job_type == JobType.TRAINING:
                # Training runs in main process (uses all GPU memory anyway)
                # Could move to subprocess for isolation, but adds complexity
                self._process_training_job(job_spec.job_id)
            else:
                # Inference runs in main process to reuse model cache
                self._process_inference_job(job_spec.job_id)
            
            job_success = True
            
        except Exception as e:
            logger.exception(f"Job {job_spec.job_id} failed: {e}")
        finally:
            self.current_job_spec = None
            self.status_reporter.set_current_job(None, None)
            self.status_reporter.record_job_completed(
                success=job_success,
                job_type=job_spec.job_type,
            )

    def run(self):
        """Main worker loop."""
        logger.info(f"Worker {self.config.worker_id} starting main loop")
        
        self.status_reporter.start()
        
        try:
            while self.running:
                try:
                    job_spec = self._claim_next_job()
                    
                    if job_spec:
                        self._process_job(job_spec)
                    else:
                        logger.debug(f"No pending jobs, waiting {self.config.poll_interval}s...")
                        time.sleep(self.config.poll_interval)
                        
                except Exception as e:
                    logger.exception(f"Error in main loop: {e}")
                    time.sleep(self.config.poll_interval)
        finally:
            self.status_reporter.stop()
            self.model_cache.clear()
        
        logger.info("Worker shutting down")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Genlio LLaMA-Factory Worker")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--worker-id", default=os.environ.get("WORKER_ID", f"worker-{uuid.uuid4().hex[:8]}"))
    parser.add_argument("--worker-name", default=os.environ.get("WORKER_NAME"), help="Human-readable worker name")
    parser.add_argument("--poll-interval", type=int, default=int(os.environ.get("POLL_INTERVAL", "5")))
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", "./data")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("OUTPUT_DIR", "./outputs")))
    parser.add_argument("--supabase-url", default=os.environ.get("SUPABASE_URL"))
    parser.add_argument("--supabase-key", default=os.environ.get("SUPABASE_KEY"))
    parser.add_argument("--model-cache-dir", type=Path, default=Path(os.environ.get("MODEL_CACHE_DIR", "./model_cache")))
    parser.add_argument("--training-min-gpu-gb", type=float, default=float(os.environ.get("TRAINING_MIN_GPU_GB", "12")))
    parser.add_argument("--inference-min-gpu-gb", type=float, default=float(os.environ.get("INFERENCE_MIN_GPU_GB", "6")))
    parser.add_argument("--inference-in-subprocess", action="store_true", default=os.environ.get("INFERENCE_IN_SUBPROCESS", "").lower() == "true")
    
    args = parser.parse_args()
    
    if not args.database_url:
        parser.error("--database-url or DATABASE_URL is required")
    if not args.supabase_url:
        parser.error("--supabase-url or SUPABASE_URL is required")
    if not args.supabase_key:
        parser.error("--supabase-key or SUPABASE_KEY is required")
    
    config = WorkerConfig(
        database_url=args.database_url,
        worker_id=args.worker_id,
        worker_name=args.worker_name,
        poll_interval=args.poll_interval,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        supabase_url=args.supabase_url,
        supabase_key=args.supabase_key,
        model_cache_dir=args.model_cache_dir,
        training_min_gpu_gb=args.training_min_gpu_gb,
        inference_min_gpu_gb=args.inference_min_gpu_gb,
        inference_in_subprocess=args.inference_in_subprocess,
    )
    
    worker = GenlioWorker(config)
    worker.run()


if __name__ == "__main__":
    main()

