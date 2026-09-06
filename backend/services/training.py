"""Training orchestration service for Vertex AI CustomJob."""

import os
import json
import time
import threading
import subprocess
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import yaml

from backend.core.config import settings
from backend.services.storage import storage_service
from backend.services.logger import operations_logger


class TrainingService:
    def __init__(self):
        self._active_jobs: Dict[str, Dict[str, Any]] = {}
        self._stop_requested: Dict[str, bool] = {}


    def launch_training(
        self,
        bucket_name: str,
        project_id: str,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        start_time = datetime.now(timezone.utc).isoformat()
        storage_service.set_active_operation(bucket_name, project_id, "TRAINING_RUNNING")
        operations_logger.log(f"Initiating training pipeline for '{project_id}'", level="INFO", source="TRAINING", project_id=project_id)

        config_path = f"{project_id}/config.yaml"
        if not storage_service.file_exists(bucket_name, config_path):
            raise FileNotFoundError(f"Missing {config_path}")

        cfg_raw = storage_service.read_file(bucket_name, config_path)
        config = yaml.safe_load(cfg_raw) or {}

        gcs_workspace = f"gs://{bucket_name}/{project_id}"
        distill_method = config.get("distillation", {}).get("method", "cot_distillation")
        hw_cfg = config.get("training", {}).get("hardware", {})
        accelerator_type = hw_cfg.get("accelerator_type", "NVIDIA_L4")
        machine_type = hw_cfg.get("machine_type", "g2-standard-8")

        # Check if Vertex AI should be invoked directly
        use_vertex = bool(settings.GCP_PROJECT_ID and not dry_run and storage_service.use_gcs)

        if use_vertex:
            try:
                from google.cloud import aiplatform

                aiplatform.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_REGION)
                job_name = f"distillfw-train-{project_id}-{int(time.time())}"

                custom_job = aiplatform.CustomJob(
                    display_name=job_name,
                    worker_pool_specs=[
                        {
                            "machine_spec": {
                                "machine_type": machine_type,
                                "accelerator_type": accelerator_type,
                                "accelerator_count": hw_cfg.get("accelerator_count", 1)
                            },
                            "replica_count": 1,
                            "container_spec": {
                                "image_uri": settings.TRAINER_IMAGE_URI,
                                "args": [
                                    f"--gcs_workspace={gcs_workspace}",
                                    f"--bucket={bucket_name}",
                                    f"--project_id={project_id}"
                                ]
                            }
                        }
                    ]
                )

                operations_logger.log(f"Submitting Vertex AI CustomJob: {job_name}", level="INFO", source="TRAINING", project_id=project_id)
                trainer_sa = settings.TRAINER_SA or (f"distillfw-trainer-sa@{settings.GCP_PROJECT_ID}.iam.gserviceaccount.com" if settings.GCP_PROJECT_ID else None)
                if trainer_sa:
                    custom_job.submit(service_account=trainer_sa)
                else:
                    custom_job.submit()
                job_id = custom_job.resource_name

                storage_service.record_history(
                    bucket_name, project_id, "TRAINING_LAUNCH", "RUNNING",
                    {"job_name": job_name, "job_id": job_id, "hardware": hw_cfg, "method": distill_method},
                    f"Submitted Vertex AI CustomJob {job_name}",
                    start_time
                )

                # Monitor Vertex AI CustomJob in background thread
                def _monitor_vertex_custom_job():
                    try:
                        operations_logger.log(f"Monitoring Vertex AI CustomJob '{job_name}'...", level="INFO", source="TRAINING", project_id=project_id)
                        while True:
                            if self._stop_requested.get(project_id):
                                try:
                                    custom_job.cancel()
                                    operations_logger.log(f"Cancelled Vertex AI CustomJob {job_name}", level="WARNING", source="TRAINING", project_id=project_id)
                                except Exception:
                                    pass
                                break

                            time.sleep(10)
                            state = custom_job.state.name if hasattr(custom_job.state, "name") else str(custom_job.state)
                            if "SUCCEEDED" in state:
                                operations_logger.log(f"Vertex AI CustomJob '{job_name}' SUCCEEDED!", level="SUCCESS", source="TRAINING", project_id=project_id)
                                storage_service.write_file(
                                    bucket_name,
                                    f"{project_id}/training/heartbeat.json",
                                    json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "status": "COMPLETED", "job_id": job_id}, indent=2)
                                )
                                storage_service.record_history(
                                    bucket_name, project_id, "TRAINING", "SUCCESS",
                                    {"job_id": job_id, "hardware": hw_cfg},
                                    f"Vertex AI CustomJob completed successfully.",
                                    start_time
                                )
                                break
                            elif "FAILED" in state or "CANCELLED" in state:
                                operations_logger.log(f"Vertex AI CustomJob '{job_name}' ended with state: {state}", level="ERROR", source="TRAINING", project_id=project_id)
                                storage_service.write_file(
                                    bucket_name,
                                    f"{project_id}/training/heartbeat.json",
                                    json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "status": "FAILED", "job_id": job_id, "error": state}, indent=2)
                                )
                                storage_service.record_history(
                                    bucket_name, project_id, "TRAINING", "FAILED",
                                    {"job_id": job_id, "error": state},
                                    f"Vertex AI CustomJob {state}.",
                                    start_time
                                )
                                break
                    except Exception as mon_err:
                        operations_logger.log(f"Vertex CustomJob monitor note: {mon_err}", level="INFO", source="TRAINING", project_id=project_id)
                    finally:
                        storage_service.set_active_operation(bucket_name, project_id, None)

                threading.Thread(target=_monitor_vertex_custom_job, daemon=True).start()
                return {"status": "SUBMITTED", "job_id": job_id, "job_name": job_name, "mode": "vertex_ai"}

            except Exception as e:
                operations_logger.log(f"Vertex AI submission note: {e}. Executing trainer pipeline locally.", level="INFO", source="TRAINING", project_id=project_id)

        # Local Execution Mode
        def run_local_training():
            try:
                operations_logger.log(f"Running training worker for '{project_id}' (Method: {distill_method})", level="INFO", source="TRAINING", project_id=project_id)
                from trainer.train import main as trainer_main
                trainer_main(
                    storage_service=storage_service,
                    custom_args=[
                        f"--gcs_workspace={gcs_workspace}",
                        f"--bucket={bucket_name}",
                        f"--project_id={project_id}",
                        "--dry_run" if dry_run else ""
                    ]
                )

                operations_logger.log(f"Training completed successfully for '{project_id}'. Saved PEFT adapter.", level="SUCCESS", source="TRAINING", project_id=project_id)
                storage_service.record_history(
                    bucket_name, project_id, "TRAINING", "SUCCESS",
                    {"distillation_method": distill_method, "hardware": hw_cfg},
                    f"Completed training run and generated PEFT adapter weights.",
                    start_time
                )
            except Exception as ex:
                operations_logger.log(f"Training worker error: {ex}", level="ERROR", source="TRAINING", project_id=project_id)
                storage_service.record_history(
                    bucket_name, project_id, "TRAINING", "FAILED",
                    {"error": str(ex)},
                    f"Training failure: {ex}",
                    start_time
                )
            finally:
                storage_service.set_active_operation(bucket_name, project_id, None)

        thread = threading.Thread(target=run_local_training, daemon=True)
        thread.start()

        return {"status": "STARTED", "mode": "local_worker", "job_id": f"local-{int(time.time())}"}

    def stop(self, bucket_name: str, project_id: str) -> Dict[str, Any]:
        self._stop_requested[project_id] = True
        storage_service.set_active_operation(bucket_name, project_id, None)
        storage_service.write_file(
            bucket_name,
            f"{project_id}/training/heartbeat.json",
            json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "status": "STOPPED"}, indent=2)
        )
        operations_logger.log(f"Training stopped by user for project '{project_id}'", level="WARNING", source="TRAINING", project_id=project_id)
        return {"status": "STOPPED", "project_id": project_id}

    def clear(self, bucket_name: str, project_id: str) -> Dict[str, Any]:
        self._stop_requested[project_id] = True
        storage_service.set_active_operation(bucket_name, project_id, None)
        storage_service.delete_file(bucket_name, f"{project_id}/training/metrics.jsonl")
        storage_service.delete_file(bucket_name, f"{project_id}/training/heartbeat.json")
        storage_service.delete_file(bucket_name, f"{project_id}/training/final_adapter/adapter_model.safetensors")
        storage_service.delete_file(bucket_name, f"{project_id}/training/final_adapter/adapter_config.json")
        operations_logger.log(f"Training artifacts cleared for '{project_id}'", level="INFO", source="TRAINING", project_id=project_id)
        return {"status": "CLEARED", "project_id": project_id}

    def get_metrics(self, bucket_name: str, project_id: str) -> List[Dict[str, Any]]:
        metrics_file = f"{project_id}/training/metrics.jsonl"
        if not storage_service.file_exists(bucket_name, metrics_file):
            return []

        content = storage_service.read_file(bucket_name, metrics_file)
        metrics = []
        for line in content.splitlines():
            if line.strip():
                try:
                    metrics.append(json.loads(line))
                except Exception:
                    pass
        return metrics

    def get_heartbeat(self, bucket_name: str, project_id: str) -> Dict[str, Any]:
        hb_file = f"{project_id}/training/heartbeat.json"
        if not storage_service.file_exists(bucket_name, hb_file):
            return {"status": "IDLE", "timestamp": None}

        try:
            return json.loads(storage_service.read_file(bucket_name, hb_file))
        except Exception:
            return {"status": "UNKNOWN", "timestamp": None}


training_service = TrainingService()

