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
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import pandas as pd
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, JSON, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base

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


@dataclass
class WorkerConfig:
    """Worker configuration."""
    database_url: str
    worker_id: str
    poll_interval: int
    data_dir: Path
    output_dir: Path
    supabase_url: str
    supabase_key: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Worker {config.worker_id} initialized")
        logger.info(f"Data directory: {config.data_dir}")
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Supabase URL: {config.supabase_url}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
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
            from llamafactory.hparams import get_train_args
            
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
            
            # Create a custom callback to report progress
            class ProgressCallback:
                def __init__(callback_self, worker, job_id):
                    callback_self.worker = worker
                    callback_self.job_id = job_id
                    callback_self.last_update = 0
                
                def on_log(callback_self, args, state, control, logs=None, **kwargs):
                    if logs:
                        metrics = {
                            "loss": logs.get("loss"),
                            "learning_rate": logs.get("learning_rate"),
                            "epoch": logs.get("epoch"),
                        }
                        callback_self.worker._update_job_status(
                            callback_self.job_id,
                            current_step=state.global_step if hasattr(state, 'global_step') else None,
                            total_steps=state.max_steps if hasattr(state, 'max_steps') else None,
                            current_epoch=logs.get("epoch"),
                            metrics=metrics,
                        )
            
            # Run training
            # Note: LLaMA-Factory's run_exp accepts a dict of args
            run_exp(args=args)
            
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
        logger.info(f"Processing job {job.job_id}: {job.job_name or 'Unnamed'}")
        
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

    def run(self):
        """Main worker loop."""
        logger.info(f"Worker {self.config.worker_id} starting main loop")
        
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

