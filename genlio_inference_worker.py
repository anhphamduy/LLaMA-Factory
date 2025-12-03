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
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
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


@dataclass
class WorkerConfig:
    """Worker configuration."""
    database_url: str
    worker_id: str
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
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Ensure cache directory exists
        config.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Inference Worker {config.worker_id} initialized")
        logger.info(f"Model cache directory: {config.model_cache_dir}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
        if self.current_job:
            self._update_job_status(
                self.current_job.job_id,
                status="failed",
                error="Worker interrupted by signal",
            )

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

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
            
            job.updated_at = self._now_iso()
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
            job.updated_at = self._now_iso()
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
        logger.info(f"Processing inference job {job.job_id}: {job.job_name or 'Unnamed'}")
        
        try:
            self._update_job_status(
                job.job_id,
                status="running",
                started_at=self._now_iso(),
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
                completed_at=self._now_iso(),
            )
            logger.info(f"Inference job {job.job_id} completed successfully")
            
        except Exception as e:
            logger.exception(f"Error processing inference job {job.job_id}: {e}")
            self._update_job_status(
                job.job_id,
                status="failed",
                error=str(e),
                completed_at=self._now_iso(),
            )
        finally:
            self.current_job = None

    def run(self):
        """Main worker loop."""
        logger.info(f"Inference Worker {self.config.worker_id} starting main loop")
        
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
        
        logger.info("Inference Worker shutting down")


def main():
    parser = argparse.ArgumentParser(description="Genlio LLaMA-Factory Inference Worker")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--worker-id", default=os.environ.get("WORKER_ID", f"infer-{uuid.uuid4().hex[:8]}"))
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
        poll_interval=args.poll_interval,
        supabase_url=args.supabase_url,
        supabase_key=args.supabase_key,
        model_cache_dir=args.model_cache_dir,
    )
    
    worker = GenlioInferenceWorker(config)
    worker.run()


if __name__ == "__main__":
    main()

