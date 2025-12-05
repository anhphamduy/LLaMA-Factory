#!/usr/bin/env python3
"""
LLaMA-Factory Worker for Genlio Integration

This worker polls the Genlio database for pending fine-tuning jobs
and executes them using LLaMA-Factory's training pipeline.

It downloads datasets directly from Supabase storage.

Usage:
    python genlio_worker.py --database-url "postgresql://..." --poll-interval 10

Environment variables:
    DATABASE_URL: PostgreSQL connection string
    WORKER_ID: Unique identifier for this worker (auto-generated if not set)
    WORKER_NAME: Human-readable name for this worker (optional)
    POLL_INTERVAL: Seconds between polls (default: 10)
    DATA_DIR: Directory for LLaMA-Factory datasets (default: ./data)
    OUTPUT_DIR: Base directory for training outputs (default: ./outputs)
    SUPABASE_URL: Supabase project URL
    SUPABASE_KEY: Supabase service key
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
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import psutil
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, JSON, ForeignKey
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
    started_at = Column(String(64), nullable=False)
    last_heartbeat = Column(String(64), nullable=False)


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
        self._jobs_completed = 0
        self._jobs_failed = 0
        self._started_at = _now_iso()
        
        # Cache static info
        self._static_info = self._collect_static_info()

    def _collect_static_info(self) -> Dict[str, Any]:
        """Collect static system information (doesn't change during runtime)."""
        info = {
            "hostname": socket.gethostname(),
            "ip_address": self._get_ip_address(),
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

    def _get_ip_address(self) -> str:
        """Get the local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _collect_gpu_metrics(self) -> tuple[int, List[Dict[str, Any]]]:
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
                
                # Get memory info (actual system memory usage, not just PyTorch)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_total_gb = mem_info.total / (1024**3)
                mem_used_gb = mem_info.used / (1024**3)
                mem_percent = (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0
                
                gpu_info = {
                    "index": i,
                    "name": name,
                    "memory_total_gb": round(mem_total_gb, 2),
                    "memory_used_gb": round(mem_used_gb, 2),
                    "memory_free_gb": round((mem_info.total - mem_info.used) / (1024**3), 2),
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
                    power = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                    gpu_info["power_watts"] = round(power / 1000, 1)
                    gpu_info["power_limit_watts"] = round(power_limit / 1000, 1)
                except Exception:
                    pass
                
                # Get fan speed (may not be available on all GPUs)
                try:
                    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                    gpu_info["fan_speed_percent"] = fan_speed
                except Exception:
                    pass
                
                # Get compute processes running on this GPU
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
            # Fall back to torch-only metrics if pynvml not available
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

    def _collect_dynamic_metrics(self) -> Dict[str, Any]:
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
        gpu_count, gpus = self._collect_gpu_metrics()
        metrics["gpu_count"] = gpu_count
        metrics["gpus"] = gpus
        
        return metrics

    def _update_status(self):
        """Update worker status in the database."""
        try:
            dynamic_metrics = self._collect_dynamic_metrics()
            
            with self.session_factory() as session:
                worker_status = session.get(WorkerStatus, self.worker_id)
                
                if worker_status is None:
                    # Create new record
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
                worker_status.jobs_completed = self._jobs_completed
                worker_status.jobs_failed = self._jobs_failed
                
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
                    worker_status.last_heartbeat = _now_iso()
                    session.commit()
        except Exception as e:
            logger.error(f"Failed to mark worker as offline: {e}")
        
        logger.info(f"Status reporter stopped for worker {self.worker_id}")

    def set_current_job(self, job_id: Optional[str], job_name: Optional[str] = None):
        """Set the current job being processed."""
        self._current_job_id = job_id
        self._current_job_name = job_name

    def record_job_completed(self, success: bool = True):
        """Record a job completion."""
        if success:
            self._jobs_completed += 1
        else:
            self._jobs_failed += 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class GenlioProgressCallback(TrainerCallback):
    """
    Custom TrainerCallback that reports training progress to the Genlio database.
    
    Records granular metrics history at each logging step for visualization:
    - Training loss at each step
    - Learning rate schedule
    - Epoch progress
    - Evaluation metrics when available
    
    Metrics are stored in the database as:
    {
        "current": { "loss": 0.5, "learning_rate": 1e-4, ... },
        "history": [
            { "step": 10, "loss": 2.3, "learning_rate": 1e-4, "epoch": 0.1, "timestamp": "..." },
            ...
        ],
        "eval_history": [
            { "step": 100, "eval_loss": 1.8, "timestamp": "..." },
            ...
        ]
    }
    """
    
    def __init__(self, worker: "GenlioWorker", job_id: str):
        super().__init__()
        self.worker = worker
        self.job_id = job_id
        self.training_started = False
        
        # Metrics history storage
        self.train_history: list[dict] = []
        self.eval_history: list[dict] = []
        
        # Throttling for DB updates (history still recorded locally)
        self.last_db_update_time = 0.0
        self.db_update_interval = 3.0  # Update DB every 3 seconds
        
        logger.info(f"[GenlioProgressCallback] CREATED for job_id={job_id}")
    
    def __repr__(self):
        return f"GenlioProgressCallback(job_id={self.job_id})"
    
    def _build_metrics_payload(self, current_metrics: dict) -> dict:
        """Build the full metrics payload with current values and history."""
        return {
            "current": current_metrics,
            "history": self.train_history,
            "eval_history": self.eval_history,
        }
    
    def _should_update_db(self) -> bool:
        """Check if enough time has passed to update DB."""
        current_time = time.time()
        if current_time - self.last_db_update_time >= self.db_update_interval:
            self.last_db_update_time = current_time
            return True
        return False
    
    def on_init_end(self, args, state, control, **kwargs):
        """Called when trainer is initialized."""
        logger.info(f"[GenlioProgressCallback] on_init_end called")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called when training starts. Captures total steps and epochs."""
        logger.info(f"[GenlioProgressCallback] on_train_begin called!")
        self.training_started = True
        self.last_db_update_time = time.time()
        
        logger.info(f"[GenlioProgressCallback] Training started: max_steps={state.max_steps}, num_epochs={args.num_train_epochs}")
        
        # Initialize metrics with empty history
        initial_metrics = self._build_metrics_payload({
            "loss": None,
            "learning_rate": None,
            "epoch": 0.0,
        })
        
        self.worker._update_job_status(
            self.job_id,
            current_step=0,
            total_steps=state.max_steps,
            current_epoch=0.0,
            total_epochs=float(args.num_train_epochs),
            metrics=initial_metrics,
        )
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when metrics are logged. Records to history and updates database."""
        logger.info(f"[GenlioProgressCallback] on_log called! logs={logs}")
        
        if not logs:
            logger.info("[GenlioProgressCallback] on_log: no logs, returning")
            return
        
        current_step = getattr(state, "global_step", 0)
        max_steps = getattr(state, "max_steps", 0)
        timestamp = _now_iso()
        
        # Extract training metrics
        loss = logs.get("loss")
        learning_rate = logs.get("learning_rate")
        epoch = logs.get("epoch")
        grad_norm = logs.get("grad_norm")
        
        # Check for eval metrics
        eval_loss = logs.get("eval_loss")
        
        # Record training metrics to history (every log event)
        if loss is not None:
            train_record = {
                "step": current_step,
                "timestamp": timestamp,
            }
            if loss is not None:
                train_record["loss"] = float(loss)
            if learning_rate is not None:
                train_record["learning_rate"] = float(learning_rate)
            if epoch is not None:
                train_record["epoch"] = float(epoch)
            if grad_norm is not None:
                train_record["grad_norm"] = float(grad_norm)
            
            self.train_history.append(train_record)
            logger.info(f"[GenlioProgressCallback] Recorded train metrics: step={current_step}, loss={loss:.4f}" if loss else f"[GenlioProgressCallback] Recorded train metrics: step={current_step}")
        
        # Record eval metrics to history
        if eval_loss is not None:
            eval_record = {
                "step": current_step,
                "timestamp": timestamp,
                "eval_loss": float(eval_loss),
            }
            # Add any other eval metrics
            for key, value in logs.items():
                if key.startswith("eval_") and key != "eval_loss" and isinstance(value, (int, float)):
                    eval_record[key] = float(value)
            
            self.eval_history.append(eval_record)
            logger.info(f"[GenlioProgressCallback] Recorded eval metrics: step={current_step}, eval_loss={eval_loss:.4f}")
        
        # Build current metrics
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
        
        # Update DB with throttling
        if self._should_update_db():
            metrics_payload = self._build_metrics_payload(current_metrics)
            
            logger.info(
                f"[GenlioProgressCallback] DB update: step={current_step}/{max_steps}, "
                f"history_len={len(self.train_history)}, eval_history_len={len(self.eval_history)}"
            )
            
            self.worker._update_job_status(
                self.job_id,
                current_step=current_step,
                total_steps=max_steps,
                current_epoch=float(epoch) if epoch is not None else None,
                metrics=metrics_payload,
            )
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        # Log every 100 steps to avoid spam
        if state.global_step % 100 == 0:
            logger.info(f"[GenlioProgressCallback] on_step_end: step={state.global_step}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends. Final update with complete history."""
        logger.info(f"[GenlioProgressCallback] on_train_end called!")
        if not self.training_started:
            logger.info("[GenlioProgressCallback] on_train_end: training not started, returning")
            return
        
        logger.info(f"[GenlioProgressCallback] Training ended: final_step={state.global_step}, "
                   f"total_train_records={len(self.train_history)}, total_eval_records={len(self.eval_history)}")
        
        # Build final current metrics from last log entry
        current_metrics = {}
        if state.log_history:
            last_log = state.log_history[-1]
            if "loss" in last_log:
                current_metrics["loss"] = float(last_log["loss"])
            if "learning_rate" in last_log:
                current_metrics["learning_rate"] = float(last_log["learning_rate"])
            if "epoch" in last_log:
                current_metrics["epoch"] = float(last_log["epoch"])
        
        # Final DB update with complete history
        metrics_payload = self._build_metrics_payload(current_metrics)
        
        self.worker._update_job_status(
            self.job_id,
            current_step=state.global_step,
            total_steps=state.max_steps,
            current_epoch=float(state.epoch) if hasattr(state, "epoch") and state.epoch else None,
            metrics=metrics_payload,
        )
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation. Records eval metrics to history."""
        logger.info(f"[GenlioProgressCallback] on_evaluate called! metrics={metrics}")
        
        if not metrics:
            return
        
        current_step = getattr(state, "global_step", 0)
        timestamp = _now_iso()
        
        # Build eval record
        eval_record = {
            "step": current_step,
            "timestamp": timestamp,
        }
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                eval_record[key] = float(value)
        
        if len(eval_record) > 2:  # Has more than just step and timestamp
            self.eval_history.append(eval_record)
            logger.info(f"[GenlioProgressCallback] Recorded evaluation: step={current_step}, metrics={eval_record}")
            
            # Force DB update after evaluation
            self.last_db_update_time = 0  # Reset to force update
            current_metrics = {"eval_step": current_step}
            current_metrics.update({k: v for k, v in eval_record.items() if k not in ["step", "timestamp"]})
            
            metrics_payload = self._build_metrics_payload(current_metrics)
            self.worker._update_job_status(
                self.job_id,
                metrics=metrics_payload,
            )


class GenlioWorker:
    """Worker that polls Genlio database and runs LLaMA-Factory training jobs."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.engine = create_engine(config.database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self.running = True
        self.current_job: Optional[FineTuningJob] = None
        
        # Supabase HTTP client for storage access
        self.http_client = httpx.Client(timeout=300)
        
        # Status reporter for monitoring
        self.status_reporter = StatusReporter(
            session_factory=self.SessionLocal,
            worker_id=config.worker_id,
            worker_type="training",
            worker_name=config.worker_name,
            heartbeat_interval=5.0,
        )
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Worker {config.worker_id} initialized")
        logger.info(f"Worker name: {config.worker_name or 'auto'}")
        logger.info(f"Data directory: {config.data_dir}")
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Supabase URL: {config.supabase_url}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
        # Stop status reporter
        self.status_reporter.stop()
        
        # If we're currently processing a job, mark it as failed
        if self.current_job:
            logger.info(f"Marking current job {self.current_job.job_id} as interrupted")
            self._update_job_status(
                self.current_job.job_id,
                status="failed",
                error="Worker interrupted by signal",
            )

    def _update_job_status(
        self,
        job_id: str,
        status: Optional[str] = None,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        current_epoch: Optional[float] = None,
        total_epochs: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        adapter_path: Optional[str] = None,
        error: Optional[str] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
    ):
        """Update job status in the database."""
        with self.SessionLocal() as session:
            job = session.get(FineTuningJob, job_id)
            if not job:
                logger.warning(f"Job {job_id} not found for status update")
                return
            
            if status is not None:
                job.status = status
            if current_step is not None:
                job.current_step = current_step
            if total_steps is not None:
                job.total_steps = total_steps
            if current_epoch is not None:
                job.current_epoch = current_epoch
            if total_epochs is not None:
                job.total_epochs = total_epochs
            if metrics is not None:
                job.metrics = metrics
            if output_dir is not None:
                job.output_dir = output_dir
            if adapter_path is not None:
                job.adapter_path = adapter_path
            if error is not None:
                job.error = error
            if started_at is not None:
                job.started_at = started_at
            if completed_at is not None:
                job.completed_at = completed_at
            
            job.updated_at = _now_iso()
            session.commit()

    def _claim_next_job(self) -> Optional[FineTuningJob]:
        """Claim the next pending job."""
        with self.SessionLocal() as session:
            job = (
                session.query(FineTuningJob)
                .filter(FineTuningJob.status == "pending")
                .order_by(FineTuningJob.created_at.asc())
                .first()
            )
            
            if not job:
                return None
            
            # Claim the job
            job.status = "queued"
            job.worker_id = self.config.worker_id
            job.updated_at = _now_iso()
            session.commit()
            
            # Refresh to get updated values
            session.refresh(job)
            
            logger.info(f"Claimed job {job.job_id}")
            return job

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
        
        # If file exists, try upsert
        if response.status_code == 400:
            response = self.http_client.put(url, headers=headers, content=file_content)
        
        response.raise_for_status()
        return storage_path

    def _upload_adapter_to_storage(self, job: FineTuningJob, local_adapter_path: str) -> str:
        """Upload adapter files to Supabase storage.
        
        Returns the storage path prefix where adapters are uploaded.
        """
        adapter_dir = Path(local_adapter_path)
        storage_prefix = f"{job.dataset_id}/{job.job_id}"
        
        # Files to upload (essential adapter files)
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
                
            # Upload essential files or any .json/.safetensors/.bin files
            should_upload = (
                file_path.name in essential_patterns or
                file_path.suffix in [".json", ".safetensors", ".bin"]
            )
            
            if should_upload:
                storage_path = f"{storage_prefix}/{file_path.name}"
                
                # Determine content type
                content_type = "application/octet-stream"
                if file_path.suffix == ".json":
                    content_type = "application/json"
                
                logger.info(f"Uploading {file_path.name} to storage...")
                
                with open(file_path, "rb") as f:
                    file_content = f.read()
                
                self._upload_to_supabase(
                    ADAPTERS_BUCKET,
                    storage_path,
                    file_content,
                    content_type,
                )
                uploaded_files.append(file_path.name)
        
        logger.info(f"Uploaded {len(uploaded_files)} adapter files to {storage_prefix}")
        return storage_prefix

    def _get_dataset_storage_path(self, job: FineTuningJob) -> Optional[str]:
        """Get the storage path for a dataset from the jobs table."""
        with self.SessionLocal() as session:
            # First, resolve the version if not specified
            version = job.version
            if not version:
                dataset = session.get(Dataset, job.dataset_id)
                if dataset:
                    version = dataset.latest_version
            
            if not version:
                logger.warning(f"Could not resolve version for dataset {job.dataset_id}")
                return None
            
            # Find the job that created this dataset version
            # Look for upload_dataset, generate_from_summaries, or deidentify_documents jobs
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
            
            # Get storage path from result_path or extra_data
            storage_path = source_job.result_path
            if not storage_path and source_job.extra_data:
                storage_path = source_job.extra_data.get("data_path")
            
            logger.info(f"Found dataset storage path: {storage_path}")
            return storage_path

    def _prepare_dataset(self, job: FineTuningJob) -> str:
        """Download and prepare dataset for training.
        
        Downloads the dataset directly from Supabase storage.
        Returns the dataset name to use in LLaMA-Factory config.
        """
        dataset_name = f"genlio_{job.job_id}"
        dataset_dir = self.config.data_dir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_file = dataset_dir / f"{dataset_name}.json"
        
        # Get the storage path from the database
        storage_path = self._get_dataset_storage_path(job)
        
        if storage_path:
            try:
                logger.info(f"Downloading dataset from Supabase: {storage_path}")
                csv_bytes = self._download_from_supabase(storage_path)
                csv_content = csv_bytes.decode("utf-8")
                
                # Parse CSV with large field support
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
                        # Add system prompt if specified and present
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
        
        dataset_info[dataset_name] = {
            "file_name": f"{dataset_name}.json"
        }
        
        with open(dataset_info_path, "w") as f:
            json.dump(dataset_info, f, indent=2)
        
        return dataset_name

    def _build_training_args(self, job: FineTuningJob, dataset_name: str) -> Dict[str, Any]:
        """Build LLaMA-Factory training arguments from job config."""
        training_config = job.training_config or {}
        lora_config = job.lora_config or {}
        dataset_config = job.dataset_config or {}
        
        output_dir = self.config.output_dir / job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation settings with defaults
        do_eval = training_config.get("do_eval", True)  # Enable eval by default
        eval_strategy = training_config.get("eval_strategy", "steps")
        eval_steps = training_config.get("eval_steps", 100)  # Eval every 100 steps by default
        val_size = training_config.get("val_size", 0.1)  # 10% validation split by default
        
        args = {
            # Model
            "model_name_or_path": job.base_model,
            "trust_remote_code": True,
            
            # Method
            "stage": job.stage,
            "do_train": True,
            "finetuning_type": job.finetuning_type,
            
            # Dataset
            "dataset": dataset_name,
            "dataset_dir": str(self.config.data_dir),
            "template": dataset_config.get("template", "llama3"),
            "cutoff_len": dataset_config.get("cutoff_len", 2048),
            "overwrite_cache": True,
            "preprocessing_num_workers": 4,
            
            # Output
            "output_dir": str(output_dir),
            "logging_steps": training_config.get("logging_steps", 10),
            "save_steps": training_config.get("save_steps", 500),
            "plot_loss": True,
            "overwrite_output_dir": True,
            
            # Training
            "per_device_train_batch_size": training_config.get("per_device_train_batch_size", 1),
            "gradient_accumulation_steps": training_config.get("gradient_accumulation_steps", 8),
            "learning_rate": training_config.get("learning_rate", 1e-4),
            "num_train_epochs": training_config.get("num_train_epochs", 3.0),
            "lr_scheduler_type": training_config.get("lr_scheduler_type", "cosine"),
            "warmup_ratio": training_config.get("warmup_ratio", 0.1),
            "max_grad_norm": training_config.get("max_grad_norm", 1.0),
            "bf16": training_config.get("bf16", True),
            "fp16": training_config.get("fp16", False),
            
            # Reporting
            "report_to": "none",
        }
        
        # Add evaluation settings if enabled
        if do_eval:
            args["do_eval"] = True
            args["eval_strategy"] = eval_strategy
            args["eval_steps"] = eval_steps
            args["val_size"] = val_size  # LLaMA-Factory will split the dataset
            args["per_device_eval_batch_size"] = training_config.get("per_device_eval_batch_size", 1)
        
        # Add LoRA config if using LoRA
        if job.finetuning_type == "lora":
            args["lora_rank"] = lora_config.get("lora_rank", 8)
            lora_alpha = lora_config.get("lora_alpha")
            if lora_alpha:
                args["lora_alpha"] = lora_alpha
            args["lora_dropout"] = lora_config.get("lora_dropout", 0.0)
            args["lora_target"] = lora_config.get("lora_target", "all")
        
        return args

    def _run_training(self, job: FineTuningJob) -> bool:
        """Run the actual training using LLaMA-Factory."""
        try:
            # Import LLaMA-Factory training function
            from llamafactory.train.tuner import run_exp
            
            # Prepare dataset
            dataset_name = self._prepare_dataset(job)
            
            # Build training arguments
            args = self._build_training_args(job, dataset_name)
            
            # Update job with output directory
            output_dir = args["output_dir"]
            self._update_job_status(
                job.job_id,
                output_dir=output_dir,
                total_epochs=args["num_train_epochs"],
            )
            
            logger.info(f"Starting training with args: {json.dumps(args, indent=2)}")
            
            # Create progress callback for reporting to database
            progress_callback = GenlioProgressCallback(
                worker=self,
                job_id=job.job_id,
            )
            
            callbacks_list = [progress_callback]
            logger.info(f"[DEBUG] Created callbacks list: {callbacks_list}")
            logger.info(f"[DEBUG] Callback types: {[type(cb).__name__ for cb in callbacks_list]}")
            logger.info(f"[DEBUG] Callback repr: {[repr(cb) for cb in callbacks_list]}")
            
            # Run training with callback
            logger.info(f"[DEBUG] Calling run_exp with callbacks={callbacks_list}")
            run_exp(args=args, callbacks=callbacks_list)
            logger.info(f"[DEBUG] run_exp completed")
            
            # Determine adapter path
            adapter_path = output_dir
            if job.finetuning_type == "lora":
                # LoRA adapters are saved in the output directory
                adapter_path = output_dir
            
            return True, adapter_path
            
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            return False, str(e)

    def _process_job(self, job: FineTuningJob):
        """Process a single fine-tuning job."""
        self.current_job = job
        self.status_reporter.set_current_job(job.job_id, job.job_name)
        logger.info(f"Processing job {job.job_id}: {job.job_name or 'Unnamed'}")
        
        job_success = False
        try:
            # Update status to running
            self._update_job_status(
                job.job_id,
                status="running",
                started_at=_now_iso(),
            )
            
            # Run training
            success, result = self._run_training(job)
            
            if success:
                local_adapter_path = result
                
                # Upload adapters to storage
                try:
                    logger.info(f"Uploading adapters to storage...")
                    storage_path = self._upload_adapter_to_storage(job, local_adapter_path)
                    adapter_path = storage_path
                    logger.info(f"Adapters uploaded to: {storage_path}")
                except Exception as upload_error:
                    logger.exception(f"Failed to upload adapters: {upload_error}")
                    # Fall back to local path if upload fails
                    adapter_path = local_adapter_path
                    logger.warning(f"Using local adapter path: {adapter_path}")
                
                self._update_job_status(
                    job.job_id,
                    status="succeeded",
                    adapter_path=adapter_path,
                    completed_at=_now_iso(),
                )
                logger.info(f"Job {job.job_id} completed successfully")
                job_success = True
            else:
                self._update_job_status(
                    job.job_id,
                    status="failed",
                    error=result,
                    completed_at=_now_iso(),
                )
                logger.error(f"Job {job.job_id} failed: {result}")
                
        except Exception as e:
            logger.exception(f"Error processing job {job.job_id}")
            self._update_job_status(
                job.job_id,
                status="failed",
                error=str(e),
                completed_at=_now_iso(),
            )
        finally:
            self.current_job = None
            self.status_reporter.set_current_job(None, None)
            self.status_reporter.record_job_completed(success=job_success)

    def run(self):
        """Main worker loop."""
        logger.info(f"Worker {self.config.worker_id} starting main loop")
        
        # Start the status reporter
        self.status_reporter.start()
        
        try:
            while self.running:
                try:
                    # Try to claim a job
                    job = self._claim_next_job()
                    
                    if job:
                        self._process_job(job)
                    else:
                        # No jobs available, wait before polling again
                        logger.debug(f"No pending jobs, waiting {self.config.poll_interval}s...")
                        time.sleep(self.config.poll_interval)
                        
                except Exception as e:
                    logger.exception(f"Error in main loop: {e}")
                    time.sleep(self.config.poll_interval)
        finally:
            # Stop the status reporter
            self.status_reporter.stop()
        
        logger.info("Worker shutting down")


def main():
    parser = argparse.ArgumentParser(description="Genlio LLaMA-Factory Worker")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection string",
    )
    parser.add_argument(
        "--worker-id",
        default=os.environ.get("WORKER_ID", str(uuid.uuid4())[:8]),
        help="Unique worker identifier",
    )
    parser.add_argument(
        "--worker-name",
        default=os.environ.get("WORKER_NAME"),
        help="Human-readable worker name",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=int(os.environ.get("POLL_INTERVAL", "10")),
        help="Seconds between polls for new jobs",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("DATA_DIR", "./data")),
        help="Directory for LLaMA-Factory datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", "./outputs")),
        help="Base directory for training outputs",
    )
    parser.add_argument(
        "--supabase-url",
        default=os.environ.get("SUPABASE_URL"),
        help="Supabase project URL",
    )
    parser.add_argument(
        "--supabase-key",
        default=os.environ.get("SUPABASE_KEY"),
        help="Supabase service key",
    )
    
    args = parser.parse_args()
    
    if not args.database_url:
        parser.error("--database-url or DATABASE_URL environment variable is required")
    
    if not args.supabase_url:
        parser.error("--supabase-url or SUPABASE_URL environment variable is required")
    
    if not args.supabase_key:
        parser.error("--supabase-key or SUPABASE_KEY environment variable is required")
    
    config = WorkerConfig(
        database_url=args.database_url,
        worker_id=args.worker_id,
        worker_name=args.worker_name,
        poll_interval=args.poll_interval,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        supabase_url=args.supabase_url,
        supabase_key=args.supabase_key,
    )
    
    worker = GenlioWorker(config)
    worker.run()


if __name__ == "__main__":
    main()

