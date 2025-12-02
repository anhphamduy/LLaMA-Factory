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


class GenlioProgressCallback(TrainerCallback):
    """
    Custom TrainerCallback that reports training progress to the Genlio database.
    
    This callback hooks into the HuggingFace Trainer lifecycle to capture:
    - Training start/end events
    - Step progress and timing
    - Loss, learning rate, and other metrics
    - Epoch progress
    
    It also records all metrics history and generates beautiful visualizations at the end.
    """
    
    def __init__(self, worker: "GenlioWorker", job_id: str, output_dir: str = None):
        super().__init__()
        self.worker = worker
        self.job_id = job_id
        self.output_dir = output_dir
        self.last_update_time = 0.0
        self.update_interval = 5.0  # Minimum seconds between DB updates to avoid flooding
        self.training_started = False
        self.start_time = None
        
        # Metrics history for tracking and visualization
        self.metrics_history = {
            "steps": [],
            "epochs": [],
            "timestamps": [],
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "grad_norm": [],
            "train_runtime": [],
            "train_samples_per_second": [],
            "perplexity": [],
        }
        
        # Evaluation history (separate for potentially different intervals)
        self.eval_history = {
            "steps": [],
            "epochs": [],
            "eval_loss": [],
            "eval_perplexity": [],
        }
        
        logger.info(f"[GenlioProgressCallback] CREATED for job_id={job_id}")
    
    def __repr__(self):
        return f"GenlioProgressCallback(job_id={self.job_id})"
    
    def on_init_end(self, args, state, control, **kwargs):
        """Called when trainer is initialized."""
        logger.info("[GenlioProgressCallback] on_init_end called")
        self.output_dir = args.output_dir
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called when training starts. Initializes tracking and sends initial update."""
        logger.info("[GenlioProgressCallback] on_train_begin called!")
        self.training_started = True
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.output_dir = args.output_dir
        
        # Reset history for fresh training run
        self.metrics_history = {
            "steps": [],
            "epochs": [],
            "timestamps": [],
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "grad_norm": [],
            "train_runtime": [],
            "train_samples_per_second": [],
            "perplexity": [],
        }
        self.eval_history = {
            "steps": [],
            "epochs": [],
            "eval_loss": [],
            "eval_perplexity": [],
        }
        
        logger.info(f"[GenlioProgressCallback] Training started: max_steps={state.max_steps}, num_epochs={args.num_train_epochs}")
        
        # Initialize with empty history structure for frontend
        initial_metrics = {
            "current": {"step": 0, "epoch": 0.0},
            "training_history": [],
            "eval_history": [],
            "summary": {
                "total_training_points": 0,
                "total_eval_points": 0,
            },
            "completed": False,
        }
        
        self.worker._update_job_status(
            self.job_id,
            current_step=0,
            total_steps=state.max_steps,
            current_epoch=0.0,
            total_epochs=float(args.num_train_epochs),
            metrics=initial_metrics,
        )
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when metrics are logged. Records all metrics and reports to database."""
        logger.info(f"[GenlioProgressCallback] on_log called! logs={logs}")
        
        if not logs:
            logger.info("[GenlioProgressCallback] on_log: no logs, returning")
            return
        
        current_time = time.time()
        current_step = getattr(state, "global_step", 0)
        max_steps = getattr(state, "max_steps", None)
        epoch = logs.get("epoch", 0.0)
        
        # Always record metrics to history (no throttling for history)
        self._record_metrics(current_step, epoch, current_time, logs)
        
        # Throttle DB updates to avoid flooding
        time_since_last = current_time - self.last_update_time
        if time_since_last < self.update_interval:
            logger.info(f"[GenlioProgressCallback] on_log: DB update throttled (only {time_since_last:.1f}s since last update)")
            return
        
        self.last_update_time = current_time
        
        # Build comprehensive metrics object with FULL HISTORY for frontend visualization
        metrics = self._build_metrics_with_history(logs)
        
        logger.info(
            f"[GenlioProgressCallback] Progress update: step={current_step}/{max_steps}, "
            f"epoch={epoch}, loss={logs.get('loss')}, lr={logs.get('learning_rate')}, "
            f"history_len={len(self.metrics_history['steps'])}"
        )
        
        self.worker._update_job_status(
            self.job_id,
            current_step=current_step,
            total_steps=max_steps,
            current_epoch=float(epoch) if epoch is not None else None,
            metrics=metrics,
        )
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        # Log every 50 steps to avoid spam
        if state.global_step % 50 == 0:
            logger.info(f"[GenlioProgressCallback] on_step_end: step={state.global_step}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation. Records eval metrics and reports to database."""
        logger.info(f"[GenlioProgressCallback] on_evaluate called! metrics={metrics}")
        
        if not metrics:
            return
        
        current_step = getattr(state, "global_step", 0)
        epoch = getattr(state, "epoch", 0.0)
        
        # Record eval metrics
        self.eval_history["steps"].append(current_step)
        self.eval_history["epochs"].append(epoch)
        
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            self.eval_history["eval_loss"].append(float(eval_loss))
            # Calculate perplexity from loss
            import math
            try:
                perplexity = math.exp(eval_loss)
                self.eval_history["eval_perplexity"].append(perplexity)
            except (OverflowError, ValueError):
                self.eval_history["eval_perplexity"].append(None)
        
        # Extract all numeric eval metrics
        eval_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                eval_metrics[key] = float(value)
        
        if eval_metrics:
            logger.info(f"[GenlioProgressCallback] Evaluation metrics: {eval_metrics}")
            self.worker._update_job_status(
                self.job_id,
                metrics=eval_metrics,
            )
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends. Generates visualizations and final update."""
        logger.info("[GenlioProgressCallback] on_train_end called!")
        
        if not self.training_started:
            logger.info("[GenlioProgressCallback] on_train_end: training not started, returning")
            return
        
        total_time = time.time() - self.start_time if self.start_time else 0
        logger.info(
            f"[GenlioProgressCallback] Training ended: final_step={state.global_step}, "
            f"total_time={total_time:.1f}s, training_points={len(self.metrics_history['steps'])}, "
            f"eval_points={len(self.eval_history['steps'])}"
        )
        
        # Generate visualizations
        try:
            self._generate_training_report(state)
        except Exception as e:
            logger.exception(f"[GenlioProgressCallback] Failed to generate training report: {e}")
        
        # Save metrics history to JSON file
        try:
            self._save_metrics_history()
        except Exception as e:
            logger.exception(f"[GenlioProgressCallback] Failed to save metrics history: {e}")
        
        # Build final comprehensive metrics with complete history
        final_metrics = self._build_final_metrics(state, total_time)
        
        self.worker._update_job_status(
            self.job_id,
            current_step=state.global_step,
            total_steps=state.max_steps,
            current_epoch=float(state.epoch) if hasattr(state, "epoch") and state.epoch else None,
            metrics=final_metrics,
        )
    
    def _record_metrics(self, step: int, epoch: float, timestamp: float, logs: dict):
        """Record metrics to history for later visualization."""
        self.metrics_history["steps"].append(step)
        self.metrics_history["epochs"].append(epoch)
        self.metrics_history["timestamps"].append(timestamp - (self.start_time or timestamp))
        
        # Training loss
        loss = logs.get("loss")
        if loss is not None:
            self.metrics_history["train_loss"].append(float(loss))
        else:
            self.metrics_history["train_loss"].append(None)
        
        # Learning rate
        lr = logs.get("learning_rate")
        if lr is not None:
            self.metrics_history["learning_rate"].append(float(lr))
        else:
            self.metrics_history["learning_rate"].append(None)
        
        # Gradient norm
        grad_norm = logs.get("grad_norm")
        if grad_norm is not None:
            self.metrics_history["grad_norm"].append(float(grad_norm))
        else:
            self.metrics_history["grad_norm"].append(None)
        
        # Eval loss (if present in training logs)
        eval_loss = logs.get("eval_loss")
        if eval_loss is not None:
            self.metrics_history["eval_loss"].append(float(eval_loss))
        else:
            self.metrics_history["eval_loss"].append(None)
        
        # Calculate perplexity from training loss
        if loss is not None:
            import math
            try:
                perplexity = math.exp(loss)
                self.metrics_history["perplexity"].append(perplexity)
            except (OverflowError, ValueError):
                self.metrics_history["perplexity"].append(None)
        else:
            self.metrics_history["perplexity"].append(None)
    
    def _extract_metrics(self, logs: dict) -> dict:
        """Extract metrics from logs for database update."""
        metrics = {}
        
        metric_keys = [
            "loss", "learning_rate", "epoch", "grad_norm",
            "eval_loss", "train_runtime", "train_samples_per_second",
        ]
        
        for key in metric_keys:
            value = logs.get(key)
            if value is not None:
                metrics[key] = float(value)
        
        return metrics
    
    def _build_metrics_with_history(self, logs: dict) -> dict:
        """Build comprehensive metrics object with full history for frontend visualization.
        
        Returns a structure optimized for frontend charting:
        {
            "current": {...},           # Latest values for quick display
            "training_history": [...],  # Array of {step, loss, lr, epoch, ...} records
            "eval_history": [...],      # Array of {step, loss, epoch, ...} records
            "summary": {...}            # Aggregate stats
        }
        """
        # Build training history as array of records (frontend-friendly format)
        training_records = []
        for i, step in enumerate(self.metrics_history["steps"]):
            record = {"step": step}
            
            if i < len(self.metrics_history["epochs"]) and self.metrics_history["epochs"][i] is not None:
                record["epoch"] = round(self.metrics_history["epochs"][i], 4)
            
            if i < len(self.metrics_history["timestamps"]) and self.metrics_history["timestamps"][i] is not None:
                record["elapsed_seconds"] = round(self.metrics_history["timestamps"][i], 1)
            
            if i < len(self.metrics_history["train_loss"]) and self.metrics_history["train_loss"][i] is not None:
                record["loss"] = round(self.metrics_history["train_loss"][i], 6)
            
            if i < len(self.metrics_history["learning_rate"]) and self.metrics_history["learning_rate"][i] is not None:
                record["learning_rate"] = self.metrics_history["learning_rate"][i]
            
            if i < len(self.metrics_history["grad_norm"]) and self.metrics_history["grad_norm"][i] is not None:
                record["grad_norm"] = round(self.metrics_history["grad_norm"][i], 6)
            
            if i < len(self.metrics_history["perplexity"]) and self.metrics_history["perplexity"][i] is not None:
                record["perplexity"] = round(self.metrics_history["perplexity"][i], 4)
            
            training_records.append(record)
        
        # Build eval history as array of records
        eval_records = []
        for i, step in enumerate(self.eval_history["steps"]):
            record = {"step": step}
            
            if i < len(self.eval_history["epochs"]) and self.eval_history["epochs"][i] is not None:
                record["epoch"] = round(self.eval_history["epochs"][i], 4)
            
            if i < len(self.eval_history["eval_loss"]) and self.eval_history["eval_loss"][i] is not None:
                record["loss"] = round(self.eval_history["eval_loss"][i], 6)
            
            if i < len(self.eval_history["eval_perplexity"]) and self.eval_history["eval_perplexity"][i] is not None:
                record["perplexity"] = round(self.eval_history["eval_perplexity"][i], 4)
            
            eval_records.append(record)
        
        # Current values (latest snapshot)
        current = self._extract_metrics(logs)
        
        # Calculate summary statistics
        train_losses = [r["loss"] for r in training_records if "loss" in r]
        summary = {
            "total_training_points": len(training_records),
            "total_eval_points": len(eval_records),
        }
        
        if train_losses:
            summary["latest_loss"] = train_losses[-1]
            summary["min_loss"] = min(train_losses)
            summary["avg_loss"] = round(sum(train_losses) / len(train_losses), 6)
        
        eval_losses = [r["loss"] for r in eval_records if "loss" in r]
        if eval_losses:
            summary["latest_eval_loss"] = eval_losses[-1]
            summary["min_eval_loss"] = min(eval_losses)
        
        return {
            "current": current,
            "training_history": training_records,
            "eval_history": eval_records,
            "summary": summary,
        }
    
    def _extract_final_metrics(self, state) -> dict:
        """Extract final metrics from trainer state."""
        metrics = {}
        
        if state.log_history:
            # Get the last few entries to find final metrics
            for log_entry in reversed(state.log_history):
                for key in ["loss", "learning_rate", "epoch", "train_loss", "eval_loss"]:
                    if key in log_entry and key not in metrics:
                        metrics[key] = float(log_entry[key])
        
        # Add summary stats
        train_losses = [x for x in self.metrics_history["train_loss"] if x is not None]
        if train_losses:
            metrics["final_train_loss"] = train_losses[-1]
            metrics["min_train_loss"] = min(train_losses)
            metrics["avg_train_loss"] = sum(train_losses) / len(train_losses)
        
        eval_losses = [x for x in self.eval_history["eval_loss"] if x is not None]
        if eval_losses:
            metrics["final_eval_loss"] = eval_losses[-1]
            metrics["min_eval_loss"] = min(eval_losses)
        
        return metrics
    
    def _build_final_metrics(self, state, total_time: float) -> dict:
        """Build final comprehensive metrics object with complete training history.
        
        This is the complete snapshot saved at training end, optimized for 
        frontend visualization with full history and summary statistics.
        """
        # Build training history as array of records
        training_records = []
        for i, step in enumerate(self.metrics_history["steps"]):
            record = {"step": step}
            
            if i < len(self.metrics_history["epochs"]) and self.metrics_history["epochs"][i] is not None:
                record["epoch"] = round(self.metrics_history["epochs"][i], 4)
            
            if i < len(self.metrics_history["timestamps"]) and self.metrics_history["timestamps"][i] is not None:
                record["elapsed_seconds"] = round(self.metrics_history["timestamps"][i], 1)
            
            if i < len(self.metrics_history["train_loss"]) and self.metrics_history["train_loss"][i] is not None:
                record["loss"] = round(self.metrics_history["train_loss"][i], 6)
            
            if i < len(self.metrics_history["learning_rate"]) and self.metrics_history["learning_rate"][i] is not None:
                record["learning_rate"] = self.metrics_history["learning_rate"][i]
            
            if i < len(self.metrics_history["grad_norm"]) and self.metrics_history["grad_norm"][i] is not None:
                record["grad_norm"] = round(self.metrics_history["grad_norm"][i], 6)
            
            if i < len(self.metrics_history["perplexity"]) and self.metrics_history["perplexity"][i] is not None:
                record["perplexity"] = round(self.metrics_history["perplexity"][i], 4)
            
            training_records.append(record)
        
        # Build eval history as array of records
        eval_records = []
        for i, step in enumerate(self.eval_history["steps"]):
            record = {"step": step}
            
            if i < len(self.eval_history["epochs"]) and self.eval_history["epochs"][i] is not None:
                record["epoch"] = round(self.eval_history["epochs"][i], 4)
            
            if i < len(self.eval_history["eval_loss"]) and self.eval_history["eval_loss"][i] is not None:
                record["loss"] = round(self.eval_history["eval_loss"][i], 6)
            
            if i < len(self.eval_history["eval_perplexity"]) and self.eval_history["eval_perplexity"][i] is not None:
                record["perplexity"] = round(self.eval_history["eval_perplexity"][i], 4)
            
            eval_records.append(record)
        
        # Calculate comprehensive summary statistics
        train_losses = [r["loss"] for r in training_records if "loss" in r]
        eval_losses = [r["loss"] for r in eval_records if "loss" in r]
        learning_rates = [r["learning_rate"] for r in training_records if "learning_rate" in r]
        grad_norms = [r["grad_norm"] for r in training_records if "grad_norm" in r]
        
        summary = {
            "total_training_points": len(training_records),
            "total_eval_points": len(eval_records),
            "total_time_seconds": round(total_time, 1),
            "final_step": state.global_step,
            "final_epoch": float(state.epoch) if hasattr(state, "epoch") and state.epoch else None,
        }
        
        if train_losses:
            summary["final_loss"] = train_losses[-1]
            summary["min_loss"] = min(train_losses)
            summary["max_loss"] = max(train_losses)
            summary["avg_loss"] = round(sum(train_losses) / len(train_losses), 6)
            # Loss improvement percentage
            if len(train_losses) > 1:
                summary["loss_improvement_pct"] = round(
                    (train_losses[0] - train_losses[-1]) / train_losses[0] * 100, 2
                )
        
        if eval_losses:
            summary["final_eval_loss"] = eval_losses[-1]
            summary["min_eval_loss"] = min(eval_losses)
            summary["best_eval_step"] = eval_records[eval_losses.index(min(eval_losses))]["step"]
        
        if learning_rates:
            summary["initial_lr"] = learning_rates[0]
            summary["final_lr"] = learning_rates[-1]
        
        if grad_norms:
            summary["avg_grad_norm"] = round(sum(grad_norms) / len(grad_norms), 6)
            summary["max_grad_norm"] = round(max(grad_norms), 6)
        
        # Current/final values for quick access
        current = {}
        if training_records:
            current = training_records[-1].copy()
        
        return {
            "current": current,
            "training_history": training_records,
            "eval_history": eval_records,
            "summary": summary,
            "completed": True,
        }
    
    def _save_metrics_history(self):
        """Save all metrics history to a JSON file."""
        if not self.output_dir:
            logger.warning("[GenlioProgressCallback] No output_dir set, skipping metrics history save")
            return
        
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        history_file = output_path / "genlio_metrics_history.json"
        
        history_data = {
            "job_id": self.job_id,
            "training_metrics": self.metrics_history,
            "evaluation_metrics": self.eval_history,
            "summary": {
                "total_steps": len(self.metrics_history["steps"]),
                "total_evals": len(self.eval_history["steps"]),
            }
        }
        
        with open(history_file, "w") as f:
            json.dump(history_data, f, indent=2, default=str)
        
        logger.info(f"[GenlioProgressCallback] Metrics history saved to {history_file}")
    
    def _generate_training_report(self, state):
        """Generate beautiful training visualizations."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            import numpy as np
        except ImportError:
            logger.warning("[GenlioProgressCallback] matplotlib not available, skipping visualization")
            return
        
        if not self.output_dir:
            logger.warning("[GenlioProgressCallback] No output_dir set, skipping visualization")
            return
        
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Filter out None values for plotting
        steps = self.metrics_history["steps"]
        train_loss = self.metrics_history["train_loss"]
        learning_rate = self.metrics_history["learning_rate"]
        grad_norm = self.metrics_history["grad_norm"]
        perplexity = self.metrics_history["perplexity"]
        
        # Create figure with custom style
        plt.style.use('dark_background')
        
        # Define color palette (vibrant, modern colors)
        colors = {
            'train_loss': '#FF6B6B',      # Coral red
            'eval_loss': '#4ECDC4',        # Teal
            'learning_rate': '#FFE66D',    # Yellow
            'grad_norm': '#95E1D3',        # Mint
            'perplexity': '#F38181',       # Salmon
            'background': '#1a1a2e',       # Dark blue-black
            'grid': '#16213e',             # Darker blue
            'text': '#eaeaea',             # Light gray
            'accent': '#e94560',           # Hot pink
        }
        
        fig = plt.figure(figsize=(16, 12), facecolor=colors['background'])
        fig.suptitle(f'🚀 Training Report - Job: {self.job_id}', 
                     fontsize=20, fontweight='bold', color=colors['text'], y=0.98)
        
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
        
        # 1. Training Loss Plot (main, larger)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor(colors['background'])
        
        valid_train = [(s, l) for s, l in zip(steps, train_loss) if l is not None]
        if valid_train:
            s_train, l_train = zip(*valid_train)
            ax1.plot(s_train, l_train, color=colors['train_loss'], linewidth=2, label='Training Loss', alpha=0.9)
            
            # Add smoothed line
            if len(l_train) > 10:
                window = min(len(l_train) // 5, 50)
                if window > 1:
                    smoothed = np.convolve(l_train, np.ones(window)/window, mode='valid')
                    ax1.plot(s_train[window-1:], smoothed, color=colors['train_loss'], 
                            linewidth=3, alpha=0.5, linestyle='--', label='Smoothed')
        
        # Add eval loss if available
        if self.eval_history["steps"] and self.eval_history["eval_loss"]:
            valid_eval = [(s, l) for s, l in zip(self.eval_history["steps"], self.eval_history["eval_loss"]) if l is not None]
            if valid_eval:
                s_eval, l_eval = zip(*valid_eval)
                ax1.scatter(s_eval, l_eval, color=colors['eval_loss'], s=100, zorder=5, 
                           label='Eval Loss', edgecolors='white', linewidth=2)
                ax1.plot(s_eval, l_eval, color=colors['eval_loss'], linewidth=2, alpha=0.7, linestyle=':')
        
        ax1.set_xlabel('Steps', fontsize=12, color=colors['text'])
        ax1.set_ylabel('Loss', fontsize=12, color=colors['text'])
        ax1.set_title('📉 Training & Evaluation Loss', fontsize=14, fontweight='bold', color=colors['text'], pad=10)
        ax1.legend(loc='upper right', facecolor=colors['grid'], edgecolor=colors['accent'])
        ax1.grid(True, alpha=0.3, color=colors['grid'])
        ax1.tick_params(colors=colors['text'])
        
        # 2. Learning Rate Schedule
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_facecolor(colors['background'])
        
        valid_lr = [(s, lr) for s, lr in zip(steps, learning_rate) if lr is not None]
        if valid_lr:
            s_lr, l_lr = zip(*valid_lr)
            ax2.plot(s_lr, l_lr, color=colors['learning_rate'], linewidth=2)
            ax2.fill_between(s_lr, l_lr, alpha=0.3, color=colors['learning_rate'])
        
        ax2.set_xlabel('Steps', fontsize=12, color=colors['text'])
        ax2.set_ylabel('Learning Rate', fontsize=12, color=colors['text'])
        ax2.set_title('📈 Learning Rate Schedule', fontsize=14, fontweight='bold', color=colors['text'], pad=10)
        ax2.grid(True, alpha=0.3, color=colors['grid'])
        ax2.tick_params(colors=colors['text'])
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # 3. Gradient Norm
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_facecolor(colors['background'])
        
        valid_gn = [(s, gn) for s, gn in zip(steps, grad_norm) if gn is not None]
        if valid_gn:
            s_gn, l_gn = zip(*valid_gn)
            ax3.plot(s_gn, l_gn, color=colors['grad_norm'], linewidth=1.5, alpha=0.7)
            
            # Add mean line
            mean_gn = sum(l_gn) / len(l_gn)
            ax3.axhline(y=mean_gn, color=colors['accent'], linestyle='--', linewidth=2, label=f'Mean: {mean_gn:.2f}')
            ax3.legend(loc='upper right', facecolor=colors['grid'], edgecolor=colors['accent'])
        
        ax3.set_xlabel('Steps', fontsize=12, color=colors['text'])
        ax3.set_ylabel('Gradient Norm', fontsize=12, color=colors['text'])
        ax3.set_title('📊 Gradient Norm', fontsize=14, fontweight='bold', color=colors['text'], pad=10)
        ax3.grid(True, alpha=0.3, color=colors['grid'])
        ax3.tick_params(colors=colors['text'])
        
        # 4. Perplexity
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.set_facecolor(colors['background'])
        
        valid_ppl = [(s, p) for s, p in zip(steps, perplexity) if p is not None and p < 1e6]  # Filter outliers
        if valid_ppl:
            s_ppl, l_ppl = zip(*valid_ppl)
            ax4.semilogy(s_ppl, l_ppl, color=colors['perplexity'], linewidth=2)
        
        ax4.set_xlabel('Steps', fontsize=12, color=colors['text'])
        ax4.set_ylabel('Perplexity (log scale)', fontsize=12, color=colors['text'])
        ax4.set_title('🎯 Perplexity', fontsize=14, fontweight='bold', color=colors['text'], pad=10)
        ax4.grid(True, alpha=0.3, color=colors['grid'])
        ax4.tick_params(colors=colors['text'])
        
        # 5. Summary Stats Box
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.set_facecolor(colors['background'])
        ax5.axis('off')
        
        # Calculate summary statistics
        stats_text = self._generate_stats_text()
        
        # Create a styled text box
        props = dict(boxstyle='round,pad=0.5', facecolor=colors['grid'], edgecolor=colors['accent'], linewidth=2)
        ax5.text(0.5, 0.5, stats_text, transform=ax5.transAxes, fontsize=12,
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace', color=colors['text'], bbox=props)
        ax5.set_title('📋 Training Summary', fontsize=14, fontweight='bold', color=colors['text'], pad=10)
        
        # Save the figure
        plot_file = output_path / "genlio_training_report.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight', facecolor=colors['background'], edgecolor='none')
        plt.close(fig)
        
        logger.info(f"[GenlioProgressCallback] Training report saved to {plot_file}")
        
        # Also generate a simple loss comparison plot
        self._generate_loss_comparison_plot(output_path, colors)
    
    def _generate_stats_text(self) -> str:
        """Generate summary statistics text for the report."""
        train_losses = [x for x in self.metrics_history["train_loss"] if x is not None]
        eval_losses = [x for x in self.eval_history["eval_loss"] if x is not None]
        
        lines = ["═" * 35]
        lines.append("       TRAINING STATISTICS")
        lines.append("═" * 35)
        
        lines.append(f"  Total Steps:      {len(self.metrics_history['steps']):>10}")
        lines.append(f"  Total Evals:      {len(self.eval_history['steps']):>10}")
        
        if train_losses:
            lines.append("")
            lines.append("  Training Loss:")
            lines.append(f"    Initial:        {train_losses[0]:>10.4f}")
            lines.append(f"    Final:          {train_losses[-1]:>10.4f}")
            lines.append(f"    Min:            {min(train_losses):>10.4f}")
            lines.append(f"    Improvement:    {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):>9.1f}%")
        
        if eval_losses:
            lines.append("")
            lines.append("  Eval Loss:")
            lines.append(f"    Best:           {min(eval_losses):>10.4f}")
            lines.append(f"    Final:          {eval_losses[-1]:>10.4f}")
        
        total_time = self.metrics_history["timestamps"][-1] if self.metrics_history["timestamps"] else 0
        lines.append("")
        lines.append(f"  Training Time:    {total_time/60:>9.1f}m")
        lines.append("═" * 35)
        
        return "\n".join(lines)
    
    def _generate_loss_comparison_plot(self, output_path: Path, colors: dict):
        """Generate a clean loss comparison plot."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=colors['background'])
        ax.set_facecolor(colors['background'])
        
        steps = self.metrics_history["steps"]
        train_loss = self.metrics_history["train_loss"]
        
        # Training loss
        valid_train = [(s, l) for s, l in zip(steps, train_loss) if l is not None]
        if valid_train:
            s_train, l_train = zip(*valid_train)
            ax.plot(s_train, l_train, color=colors['train_loss'], linewidth=2, 
                   label='Training Loss', alpha=0.8)
        
        # Eval loss
        if self.eval_history["steps"] and self.eval_history["eval_loss"]:
            valid_eval = [(s, l) for s, l in zip(self.eval_history["steps"], self.eval_history["eval_loss"]) if l is not None]
            if valid_eval:
                s_eval, l_eval = zip(*valid_eval)
                ax.plot(s_eval, l_eval, color=colors['eval_loss'], linewidth=2.5, 
                       label='Eval Loss', marker='o', markersize=8)
        
        ax.set_xlabel('Steps', fontsize=14, color=colors['text'])
        ax.set_ylabel('Loss', fontsize=14, color=colors['text'])
        ax.set_title(f'Training Progress - {self.job_id}', fontsize=16, fontweight='bold', color=colors['text'])
        ax.legend(loc='upper right', facecolor=colors['grid'], edgecolor=colors['accent'], fontsize=12)
        ax.grid(True, alpha=0.3, color=colors['grid'])
        ax.tick_params(colors=colors['text'], labelsize=12)
        
        for spine in ax.spines.values():
            spine.set_color(colors['grid'])
        
        plot_file = output_path / "genlio_loss_curve.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight', facecolor=colors['background'], edgecolor='none')
        plt.close(fig)
        
        logger.info(f"[GenlioProgressCallback] Loss curve saved to {plot_file}")


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

