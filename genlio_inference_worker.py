#!/usr/bin/env python3
"""
LLaMA-Factory Inference Worker for Genlio Integration

This worker polls the Genlio database for pending inference jobs
and executes them using LLaMA-Factory's chat model.

Separate from the training worker to allow parallel operation.

Usage:
    python genlio_inference_worker.py --database-url "postgresql://..." --poll-interval 5

Environment variables:
    DATABASE_URL: PostgreSQL connection string
    WORKER_ID: Unique identifier for this worker (auto-generated if not set)
    WORKER_NAME: Human-readable name for this worker (optional)
    POLL_INTERVAL: Seconds between polls (default: 5)
    SUPABASE_URL: Supabase project URL
    SUPABASE_KEY: Supabase service key
    MODEL_CACHE_DIR: Directory to cache downloaded adapters (default: ./model_cache)
"""
from __future__ import annotations

import argparse
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
import psutil
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, JSON
from sqlalchemy.orm import sessionmaker, declarative_base

# Add LLaMA-Factory to path
LLAMA_FACTORY_ROOT = Path(__file__).parent
sys.path.insert(0, str(LLAMA_FACTORY_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("genlio_inference_worker")

ADAPTERS_BUCKET = "adapters"

Base = declarative_base()


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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


@dataclass
class WorkerConfig:
    """Worker configuration."""
    database_url: str
    worker_id: str
    worker_name: Optional[str]
    poll_interval: int
    supabase_url: str
    supabase_key: str
    model_cache_dir: Path


class ModelCache:
    """Cache for loaded models to avoid reloading for consecutive requests."""
    
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
        
        # Unload previous model
        if self.chat_model is not None:
            logger.info(f"Unloading previous model: {self.current_model_key}")
            del self.chat_model
            self.chat_model = None
            torch_gc()
        
        # Build args for ChatModel
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


class GenlioInferenceWorker:
    """Worker that polls Genlio database and runs inference jobs."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.engine = create_engine(config.database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self.running = True
        self.current_job: Optional[InferenceJob] = None
        self.model_cache = ModelCache()
        
        # Supabase HTTP client
        self.http_client = httpx.Client(timeout=300)
        
        # Status reporter for monitoring
        self.status_reporter = StatusReporter(
            session_factory=self.SessionLocal,
            worker_id=config.worker_id,
            worker_type="inference",
            worker_name=config.worker_name,
            heartbeat_interval=5.0,
        )
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Ensure cache directory exists
        config.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Inference Worker {config.worker_id} initialized")
        logger.info(f"Worker name: {config.worker_name or 'auto'}")
        logger.info(f"Model cache directory: {config.model_cache_dir}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
        # Stop status reporter
        self.status_reporter.stop()
        
        if self.current_job:
            self._update_job_status(
                self.current_job.job_id,
                status="failed",
                error="Worker interrupted by signal",
            )

    def _update_job_status(
        self,
        job_id: str,
        status: Optional[str] = None,
        output: Optional[Dict[str, Any]] = None,
        rows_processed: Optional[int] = None,
        rows_total: Optional[int] = None,
        error: Optional[str] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
    ):
        """Update job status in the database."""
        with self.SessionLocal() as session:
            job = session.get(InferenceJob, job_id)
            if not job:
                logger.warning(f"Job {job_id} not found for status update")
                return
            
            if status is not None:
                job.status = status
            if output is not None:
                job.output = output
            if rows_processed is not None:
                job.rows_processed = rows_processed
            if rows_total is not None:
                job.rows_total = rows_total
            if error is not None:
                job.error = error
            if started_at is not None:
                job.started_at = started_at
            if completed_at is not None:
                job.completed_at = completed_at
            
            job.updated_at = _now_iso()
            session.commit()

    def _claim_next_job(self) -> Optional[InferenceJob]:
        """Claim the next pending job."""
        with self.SessionLocal() as session:
            job = (
                session.query(InferenceJob)
                .filter(InferenceJob.status == "pending")
                .order_by(InferenceJob.created_at.asc())
                .first()
            )
            
            if not job:
                return None
            
            job.status = "queued"
            job.worker_id = self.config.worker_id
            job.updated_at = _now_iso()
            session.commit()
            session.refresh(job)
            
            logger.info(f"Claimed inference job {job.job_id}")
            return job

    def _download_adapter(self, storage_path: str) -> Path:
        """Download adapter files from Supabase storage."""
        local_dir = self.config.model_cache_dir / storage_path.replace("/", "_")
        
        # Check if already cached
        adapter_config = local_dir / "adapter_config.json"
        if adapter_config.exists():
            logger.info(f"Using cached adapter: {local_dir}")
            return local_dir
        
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # List and download adapter files
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
        
        # Convert messages to LLaMA-Factory format
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

    def _run_batch_inference(self, job: InferenceJob, chat_model: Any) -> Dict[str, Any]:
        """Run batch inference on a test dataset."""
        # Download test dataset
        # For now, assume prompts are provided in job.messages as a list
        # In production, you'd download from Supabase storage
        
        results = []
        gen_config = job.generation_config or {}
        system = job.system_prompt
        total_latency = 0
        
        # Get prompts from test dataset (simplified - you'd fetch from storage)
        prompts = job.messages or []  # Each message is a prompt to test
        
        self._update_job_status(job.job_id, rows_total=len(prompts))
        
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
                
                # Add expected if available
                if job.expected_column and isinstance(prompt_data, dict):
                    result["expected"] = prompt_data.get("expected", "")
                
                results.append(result)
                
                # Update progress
                self._update_job_status(job.job_id, rows_processed=i + 1)
                
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

    def _process_job(self, job: InferenceJob):
        """Process a single inference job."""
        self.current_job = job
        self.status_reporter.set_current_job(job.job_id, job.job_name)
        logger.info(f"Processing inference job {job.job_id}: {job.job_name or 'Unnamed'}")
        
        job_success = False
        try:
            self._update_job_status(
                job.job_id,
                status="running",
                started_at=_now_iso(),
            )
            
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
                output = self._run_batch_inference(job, chat_model)
            
            self._update_job_status(
                job.job_id,
                status="succeeded",
                output=output,
                completed_at=_now_iso(),
            )
            logger.info(f"Inference job {job.job_id} completed successfully")
            job_success = True
            
        except Exception as e:
            logger.exception(f"Error processing inference job {job.job_id}: {e}")
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
        logger.info(f"Inference Worker {self.config.worker_id} starting main loop")
        
        # Start the status reporter
        self.status_reporter.start()
        
        try:
            while self.running:
                try:
                    job = self._claim_next_job()
                    
                    if job:
                        self._process_job(job)
                    else:
                        logger.debug(f"No pending jobs, waiting {self.config.poll_interval}s...")
                        time.sleep(self.config.poll_interval)
                        
                except Exception as e:
                    logger.exception(f"Error in main loop: {e}")
                    time.sleep(self.config.poll_interval)
        finally:
            # Stop the status reporter
            self.status_reporter.stop()
        
        logger.info("Inference Worker shutting down")


def main():
    parser = argparse.ArgumentParser(description="Genlio LLaMA-Factory Inference Worker")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--worker-id", default=os.environ.get("WORKER_ID", f"infer-{uuid.uuid4().hex[:8]}"))
    parser.add_argument("--worker-name", default=os.environ.get("WORKER_NAME"), help="Human-readable worker name")
    parser.add_argument("--poll-interval", type=int, default=int(os.environ.get("POLL_INTERVAL", "5")))
    parser.add_argument("--supabase-url", default=os.environ.get("SUPABASE_URL"))
    parser.add_argument("--supabase-key", default=os.environ.get("SUPABASE_KEY"))
    parser.add_argument("--model-cache-dir", type=Path, default=Path(os.environ.get("MODEL_CACHE_DIR", "./model_cache")))
    
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
        supabase_url=args.supabase_url,
        supabase_key=args.supabase_key,
        model_cache_dir=args.model_cache_dir,
    )
    
    worker = GenlioInferenceWorker(config)
    worker.run()


if __name__ == "__main__":
    main()



