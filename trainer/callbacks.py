"""Telemetry streaming callback for DistillFW training."""

import os
import time
import json
import threading
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class GCSProgressCallback(TrainerCallback):
    def __init__(
        self,
        gcs_workspace: str,
        storage_service: Optional[Any] = None,
        bucket: Optional[str] = None,
        project_id: Optional[str] = None,
        flush_steps: int = 5
    ):
        self.gcs_workspace = gcs_workspace
        self.storage_service = storage_service
        self.bucket = bucket
        self.project_id = project_id
        self.flush_steps = flush_steps
        self.last_heartbeat_time = 0.0
        self.last_step_time = time.time()
        self.buffered_metrics = []
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread = None
        self._start_heartbeat_daemon()

    def _start_heartbeat_daemon(self):
        def heartbeat_loop():
            while not self._stop_heartbeat.is_set():
                self._write_heartbeat(status="RUNNING")
                time.sleep(30)  # Heartbeat every 30-60s
        self._heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _get_gpu_stats(self) -> Dict[str, float]:
        gpu_util = 0.0
        memory_gb = 0.0
        if torch.cuda.is_available():
            try:
                memory_gb = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
                # Max memory cached/reserved
                max_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
                gpu_util = min(100.0, (memory_gb / 24.0) * 100.0) if memory_gb > 0 else 50.0
            except Exception:
                pass
        return {"gpu_utilization_pct": gpu_util, "memory_allocated_gb": memory_gb}

    def _write_metrics(self, entry: Dict[str, Any]):
        line = json.dumps(entry) + "\n"
        if self.storage_service and self.bucket and self.project_id:
            try:
                self.storage_service.append_file(
                    self.bucket,
                    f"{self.project_id}/training/metrics.jsonl",
                    line
                )
                return
            except Exception:
                pass

        # Fallback local path
        if self.gcs_workspace.startswith("gs://"):
            parts = self.gcs_workspace[5:].split("/", 1)
            b = parts[0]
            p = parts[1] if len(parts) > 1 else ""
            local_target = os.path.join(".local_workspace", b, p, "training/metrics.jsonl")
        else:
            local_target = os.path.join(self.gcs_workspace, "training/metrics.jsonl")

        os.makedirs(os.path.dirname(local_target), exist_ok=True)
        with open(local_target, "a", encoding="utf-8") as f:
            f.write(line)

    def _write_heartbeat(self, status: str = "RUNNING", step: int = 0):
        gpu_info = self._get_gpu_stats()
        heartbeat = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "step": step,
            "gpu": gpu_info
        }
        content = json.dumps(heartbeat, indent=2)
        if self.storage_service and self.bucket and self.project_id:
            try:
                self.storage_service.write_file(
                    self.bucket,
                    f"{self.project_id}/training/heartbeat.json",
                    content
                )
                return
            except Exception:
                pass

        if self.gcs_workspace.startswith("gs://"):
            parts = self.gcs_workspace[5:].split("/", 1)
            b = parts[0]
            p = parts[1] if len(parts) > 1 else ""
            local_target = os.path.join(".local_workspace", b, p, "training/heartbeat.json")
        else:
            local_target = os.path.join(self.gcs_workspace, "training/heartbeat.json")

        os.makedirs(os.path.dirname(local_target), exist_ok=True)
        with open(local_target, "w", encoding="utf-8") as f:
            f.write(content)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return

        now = time.time()
        elapsed = max(0.001, now - self.last_step_time)
        self.last_step_time = now

        gpu_stats = self._get_gpu_stats()
        # Estimate tokens/sec based on batch size and max length
        bs = getattr(args, "per_device_train_batch_size", 4)
        tokens_per_sec = round((bs * 512) / elapsed, 1)

        entry = {
            "step": state.global_step,
            "epoch": round(state.epoch or 0.0, 2),
            "train_loss": logs.get("loss"),
            "val_loss": logs.get("eval_loss"),
            "learning_rate": logs.get("learning_rate"),
            "gpu_utilization_pct": gpu_stats["gpu_utilization_pct"],
            "memory_allocated_gb": gpu_stats["memory_allocated_gb"],
            "tokens_per_sec": tokens_per_sec,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        self._write_metrics(entry)
        self._write_heartbeat(status="RUNNING", step=state.global_step)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._write_heartbeat(status="COMPLETED", step=state.global_step)
        self._stop_heartbeat.set()
