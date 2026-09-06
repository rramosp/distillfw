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

                staging_bucket_uri = f"gs://{bucket_name}"
                aiplatform.init(
                    project=settings.GCP_PROJECT_ID,
                    location=settings.GCP_REGION,
                    staging_bucket=staging_bucket_uri
                )
                job_name = f"distillfw-train-{project_id}-{int(time.time())}"

                custom_job = aiplatform.CustomJob(
                    display_name=job_name,
                    staging_bucket=staging_bucket_uri,
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
                
                try:
                    if trainer_sa:
                        custom_job.submit(service_account=trainer_sa)
                    else:
                        custom_job.submit()
                except Exception as submit_sa_err:
                    operations_logger.log(
                        f"CustomJob submit with SA '{trainer_sa}' notice ({submit_sa_err}). Retrying without explicit SA...",
                        level="WARNING",
                        source="TRAINING",
                        project_id=project_id
                    )
                    custom_job.submit()

                job_id = custom_job.resource_name
                web_url = f"https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={settings.GCP_PROJECT_ID}"

                # Write initial RUNNING heartbeat immediately upon submission
                storage_service.write_file(
                    bucket_name,
                    f"{project_id}/training/heartbeat.json",
                    json.dumps({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "status": "RUNNING",
                        "job_id": job_id,
                        "job_name": job_name,
                        "mode": "vertex_ai",
                        "web_url": web_url,
                        "machine_type": machine_type,
                        "accelerator_type": accelerator_type
                    }, indent=2)
                )

                storage_service.record_history(
                    bucket_name, project_id, "TRAINING_LAUNCH", "RUNNING",
                    {"job_name": job_name, "job_id": job_id, "hardware": hw_cfg, "method": distill_method, "web_url": web_url},
                    f"Submitted Vertex AI CustomJob {job_name} ({job_id})",
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
                            try:
                                # Fetch latest job status from Vertex AI API
                                custom_job._gca_resource = custom_job.api_client.get_custom_job(name=custom_job.resource_name)
                            except Exception:
                                pass

                            state = custom_job.state.name if hasattr(custom_job.state, "name") else str(custom_job.state)
                            if "SUCCEEDED" in state:
                                operations_logger.log(f"Vertex AI CustomJob '{job_name}' SUCCEEDED!", level="SUCCESS", source="TRAINING", project_id=project_id)
                                storage_service.write_file(
                                    bucket_name,
                                    f"{project_id}/training/heartbeat.json",
                                    json.dumps({
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "status": "COMPLETED",
                                        "job_id": job_id,
                                        "job_name": job_name,
                                        "web_url": web_url,
                                        "mode": "vertex_ai"
                                    }, indent=2)
                                )
                                storage_service.record_history(
                                    bucket_name, project_id, "TRAINING", "SUCCESS",
                                    {"job_id": job_id, "job_name": job_name, "hardware": hw_cfg},
                                    f"Vertex AI CustomJob completed successfully.",
                                    start_time
                                )
                                break
                            elif "FAILED" in state or "CANCELLED" in state:
                                err_detail = getattr(custom_job._gca_resource, "error", None)
                                err_text = str(err_detail.message) if err_detail and getattr(err_detail, "message", None) else state
                                operations_logger.log(f"Vertex AI CustomJob '{job_name}' ended with state: {state} ({err_text})", level="ERROR", source="TRAINING", project_id=project_id)
                                storage_service.write_file(
                                    bucket_name,
                                    f"{project_id}/training/heartbeat.json",
                                    json.dumps({
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "status": "FAILED",
                                        "job_id": job_id,
                                        "job_name": job_name,
                                        "web_url": web_url,
                                        "error": err_text,
                                        "mode": "vertex_ai"
                                    }, indent=2)
                                )
                                storage_service.record_history(
                                    bucket_name, project_id, "TRAINING", "FAILED",
                                    {"job_id": job_id, "error": err_text},
                                    f"Vertex AI CustomJob {state}: {err_text}",
                                    start_time
                                )
                                break
                            else:
                                # Update timestamp on running heartbeat
                                try:
                                    cur_hb = self.get_heartbeat(bucket_name, project_id)
                                    cur_hb["timestamp"] = datetime.now(timezone.utc).isoformat()
                                    cur_hb["status"] = "RUNNING"
                                    storage_service.write_file(
                                        bucket_name,
                                        f"{project_id}/training/heartbeat.json",
                                        json.dumps(cur_hb, indent=2)
                                    )
                                except Exception:
                                    pass
                    except Exception as mon_err:
                        operations_logger.log(f"Vertex CustomJob monitor note: {mon_err}", level="INFO", source="TRAINING", project_id=project_id)
                    finally:
                        storage_service.set_active_operation(bucket_name, project_id, None)

                threading.Thread(target=_monitor_vertex_custom_job, daemon=True).start()
                return {"status": "SUBMITTED", "job_id": job_id, "job_name": job_name, "web_url": web_url, "mode": "vertex_ai"}

            except Exception as e:
                err_str = f"Failed to submit Vertex AI CustomJob: {e}"
                operations_logger.log(err_str, level="ERROR", source="TRAINING", project_id=project_id)
                storage_service.write_file(
                    bucket_name,
                    f"{project_id}/training/heartbeat.json",
                    json.dumps({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "status": "FAILED",
                        "error": str(e),
                        "mode": "vertex_ai"
                    }, indent=2)
                )
                storage_service.record_history(
                    bucket_name, project_id, "TRAINING", "FAILED",
                    {"error": str(e)},
                    err_str,
                    start_time
                )
                storage_service.set_active_operation(bucket_name, project_id, None)
                raise RuntimeError(err_str)

        # Local Execution Mode (Dry-run or local fallback when explicitly requested)
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
        
        # Read heartbeat to see if there is an active Vertex AI CustomJob
        hb = self.get_heartbeat(bucket_name, project_id)
        job_id = hb.get("job_id")
        cancelled_vertex = False
        if job_id and settings.GCP_PROJECT_ID and "customJobs" in str(job_id):
            try:
                from google.cloud import aiplatform
                aiplatform.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_REGION)
                job = aiplatform.CustomJob.get(resource_name=job_id)
                job.cancel()
                cancelled_vertex = True
                operations_logger.log(f"Cancelled Vertex AI CustomJob '{job_id}'", level="WARNING", source="TRAINING", project_id=project_id)
            except Exception as ce:
                operations_logger.log(f"Notice cancelling Vertex job {job_id}: {ce}", level="INFO", source="TRAINING", project_id=project_id)

        # Also search for any active CustomJobs matching project prefix on Vertex AI
        if settings.GCP_PROJECT_ID:
            try:
                from google.cloud import aiplatform
                aiplatform.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_REGION)
                prefix = f"distillfw-train-{project_id}"
                for cj in aiplatform.CustomJob.list():
                    disp = cj.display_name or ""
                    if prefix in disp and str(cj.state) in ["1", "2", "3", "JOB_STATE_RUNNING", "JOB_STATE_PENDING", "JOB_STATE_QUEUED"]:
                        try:
                            cj.cancel()
                            cancelled_vertex = True
                            operations_logger.log(f"Cancelled active Vertex CustomJob '{disp}'", level="WARNING", source="TRAINING", project_id=project_id)
                        except Exception:
                            pass
            except Exception:
                pass

        storage_service.write_file(
            bucket_name,
            f"{project_id}/training/heartbeat.json",
            json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "STOPPED",
                "job_id": job_id
            }, indent=2)
        )
        operations_logger.log(f"Training stopped by user for project '{project_id}'", level="WARNING", source="TRAINING", project_id=project_id)
        return {"status": "STOPPED", "project_id": project_id, "cancelled_vertex": cancelled_vertex}

    def clear(self, bucket_name: str, project_id: str) -> Dict[str, Any]:
        self.stop(bucket_name, project_id)
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

