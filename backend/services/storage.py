"""Storage service for Google Cloud Storage and local emulation."""

import os
import json
import yaml
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError

from backend.core.config import settings
from backend.core.models import ProjectStatus
from backend.services.logger import operations_logger


class StorageService:
    def __init__(self):
        self._gcs_client: Optional[storage.Client] = None
        self.use_gcs: bool = False
        self._init_client()

    def _init_client(self):
        if settings.STORAGE_MODE == "local":
            self.use_gcs = False
            return

        try:
            self._gcs_client = storage.Client(project=settings.GCP_PROJECT_ID or None)
            self.use_gcs = True
        except (DefaultCredentialsError, Exception) as e:
            operations_logger.log(
                f"GCS client initialization skipped ({str(e)}). Falling back to local workspace emulation.",
                level="WARNING",
                source="STORAGE"
            )
            self.use_gcs = False

    def get_local_path(self, bucket: str, path: str = "") -> str:
        base = os.path.join(settings.LOCAL_STORAGE_ROOT, bucket)
        if path:
            return os.path.join(base, path.lstrip("/"))
        return base

    # ---------------- Bucket and Project Management ----------------

    def list_buckets(self) -> List[str]:
        buckets = set()
        default_b = settings.DEFAULT_BUCKET
        buckets.add(default_b)

        if self.use_gcs and self._gcs_client:
            try:
                for b in self._gcs_client.list_buckets():
                    buckets.add(b.name)
            except Exception as e:
                operations_logger.log(f"Failed to list GCS buckets: {e}", level="WARNING", source="STORAGE")

        # Also check local storage directory
        if os.path.exists(settings.LOCAL_STORAGE_ROOT):
            for item in os.listdir(settings.LOCAL_STORAGE_ROOT):
                if os.path.isdir(os.path.join(settings.LOCAL_STORAGE_ROOT, item)):
                    buckets.add(item)

        return sorted(list(buckets))

    def create_bucket_if_not_exists(self, bucket_name: str) -> bool:
        if self.use_gcs and self._gcs_client:
            try:
                bucket = self._gcs_client.bucket(bucket_name)
                if not bucket.exists():
                    kwargs = {"location": settings.GCP_REGION}
                    if settings.GCP_PROJECT_ID:
                        kwargs["project"] = settings.GCP_PROJECT_ID
                    self._gcs_client.create_bucket(bucket, **kwargs)
                    operations_logger.log(f"Created GCS bucket '{bucket_name}'", level="SUCCESS", source="STORAGE")
                return True
            except Exception as e:
                operations_logger.log(f"Error creating GCS bucket '{bucket_name}': {e}", level="ERROR", source="STORAGE")

        # Ensure local fallback directory exists
        local_dir = self.get_local_path(bucket_name)
        os.makedirs(local_dir, exist_ok=True)
        return True

    def list_projects(self, bucket_name: str) -> List[str]:
        projects = set()
        prefix_len = 0

        if self.use_gcs and self._gcs_client:
            try:
                bucket = self._gcs_client.bucket(bucket_name)
                if bucket.exists():
                    iterator = bucket.list_blobs(delimiter="/")
                    for page in iterator.pages:
                        for prefix in page.prefixes:
                            clean = prefix.rstrip("/")
                            if clean:
                                projects.add(clean)
            except Exception as e:
                operations_logger.log(f"Error listing projects in GCS bucket '{bucket_name}': {e}", level="WARNING", source="STORAGE")

        local_dir = self.get_local_path(bucket_name)
        if os.path.exists(local_dir):
            for item in os.listdir(local_dir):
                if os.path.isdir(os.path.join(local_dir, item)):
                    projects.add(item)

        return sorted(list(projects))

    def create_project(self, bucket_name: str, project_id: str, description: Optional[str] = None) -> Dict[str, Any]:
        self.create_bucket_if_not_exists(bucket_name)
        
        # Check if project already exists
        projects = self.list_projects(bucket_name)
        if project_id not in projects:
            # Create default directories and status
            self.write_file(bucket_name, f"{project_id}/status.json", json.dumps({
                "project_id": project_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "description": description or f"Distillation project {project_id}"
            }, indent=2))

            self.record_history(bucket_name, project_id, "PROJECT_INITIALIZATION", "SUCCESS", {
                "project_id": project_id,
                "description": description
            }, "Created project workspace structure.")

            operations_logger.log(f"Project '{project_id}' initialized in bucket '{bucket_name}'", level="SUCCESS", source="WORKSPACE", project_id=project_id)

        return {"project_id": project_id, "bucket": bucket_name, "status": "INITIALIZED"}

    # ---------------- File Read / Write / Exists ----------------

    def file_exists(self, bucket_name: str, relative_path: str) -> bool:
        if self.use_gcs and self._gcs_client:
            try:
                bucket = self._gcs_client.bucket(bucket_name)
                blob = bucket.blob(relative_path)
                if blob.exists():
                    return True
            except Exception:
                pass

        local_file = self.get_local_path(bucket_name, relative_path)
        return os.path.exists(local_file)

    def read_file(self, bucket_name: str, relative_path: str) -> str:
        if self.use_gcs and self._gcs_client:
            try:
                bucket = self._gcs_client.bucket(bucket_name)
                blob = bucket.blob(relative_path)
                if blob.exists():
                    return blob.download_as_text()
            except Exception as e:
                operations_logger.log(f"GCS read error on '{relative_path}': {e}", level="WARNING", source="STORAGE")

        local_file = self.get_local_path(bucket_name, relative_path)
        if os.path.exists(local_file):
            with open(local_file, "r", encoding="utf-8") as f:
                return f.read()
        raise FileNotFoundError(f"File '{relative_path}' not found in bucket '{bucket_name}'")

    def write_file(self, bucket_name: str, relative_path: str, content: str) -> None:
        if self.use_gcs and self._gcs_client:
            try:
                bucket = self._gcs_client.bucket(bucket_name)
                blob = bucket.blob(relative_path)
                blob.upload_from_string(content)
                # Keep local copy synced if directory exists
            except Exception as e:
                operations_logger.log(f"GCS write error on '{relative_path}': {e}", level="WARNING", source="STORAGE")

        local_file = self.get_local_path(bucket_name, relative_path)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        with open(local_file, "w", encoding="utf-8") as f:
            f.write(content)

    def append_file(self, bucket_name: str, relative_path: str, content: str) -> None:
        existing = ""
        if self.file_exists(bucket_name, relative_path):
            try:
                existing = self.read_file(bucket_name, relative_path)
            except Exception:
                existing = ""
        self.write_file(bucket_name, relative_path, existing + content)

    def delete_file(self, bucket_name: str, relative_path: str) -> None:
        if self.use_gcs and self._gcs_client:
            try:
                bucket = self._gcs_client.bucket(bucket_name)
                blob = bucket.blob(relative_path)
                if blob.exists():
                    blob.delete()
            except Exception as e:
                operations_logger.log(f"GCS delete error on '{relative_path}': {e}", level="WARNING", source="STORAGE")

        local_file = self.get_local_path(bucket_name, relative_path)
        if os.path.exists(local_file):
            os.remove(local_file)

    # ---------------- History Tracking ----------------

    def get_history(self, bucket_name: str, project_id: str) -> List[Dict[str, Any]]:
        path = f"{project_id}/history.json"
        if not self.file_exists(bucket_name, path):
            return []
        try:
            raw = self.read_file(bucket_name, path)
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def record_history(
        self,
        bucket_name: str,
        project_id: str,
        action: str,
        status: str,
        parameters: Optional[Dict[str, Any]] = None,
        details: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        entry = {
            "id": f"{action}_{int(datetime.now(timezone.utc).timestamp())}",
            "action": action,
            "status": status,
            "start_time": start_time or now,
            "end_time": end_time or now,
            "parameters": parameters or {},
            "details": details or ""
        }
        history = self.get_history(bucket_name, project_id)
        history.append(entry)
        self.write_file(bucket_name, f"{project_id}/history.json", json.dumps(history, indent=2))

    # ---------------- Deterministic Status Inference ----------------

    def infer_status(self, bucket_name: str, project_id: str) -> Dict[str, Any]:
        """
        Derives current project state deterministically from GCS artifacts:
        1. UNINITIALIZED: Missing config.yaml.
        2. CONFIGURED: config.yaml exists, but no dataset present.
        3. DATASET_READY: data/split_dataset.jsonl exists and is validated.
        4. TEACHER_INFERENCE_RUNNING: Inference job active.
        5. TEACHER_INFERENCE_DONE: data/teacher_inferences.jsonl exists.
        6. COST_ESTIMATED: cost/cost_estimate.json exists.
        7. TRAINING_RUNNING: Vertex CustomJob active, training/metrics.jsonl updating.
        8. TRAINING_COMPLETED: training/final_adapter/adapter_model.safetensors exists.
        9. EVALUATING: Evaluation job active on test split.
        10. EVALUATED: evaluation/eval_results.json exists.
        11. DEPLOYED: deployment/endpoint_metadata.json exists with an active Vertex AI Endpoint.
        """
        p = f"{project_id}/"

        # Check in-flight operation flags from status.json if present
        running_flag = None
        status_file = f"{p}status.json"
        if self.file_exists(bucket_name, status_file):
            try:
                s_data = json.loads(self.read_file(bucket_name, status_file))
                running_flag = s_data.get("active_operation")
            except Exception:
                pass

        if running_flag == "TEACHER_INFERENCE_RUNNING":
            return {"status": ProjectStatus.TEACHER_INFERENCE_RUNNING.value, "detail": "Teacher inference in progress"}
        if running_flag == "TRAINING_RUNNING":
            return {"status": ProjectStatus.TRAINING_RUNNING.value, "detail": "Model training job active"}
        if running_flag == "EVALUATING":
            return {"status": ProjectStatus.EVALUATING.value, "detail": "Evaluation running on test split"}
        if running_flag == "DEPLOYING":
            return {"status": ProjectStatus.DEPLOYING.value, "detail": "Deploying vLLM serving container to Vertex AI Endpoint"}

        # Check endpoints and deployment
        if self.file_exists(bucket_name, f"{p}deployment/endpoint_metadata.json"):
            try:
                meta = json.loads(self.read_file(bucket_name, f"{p}deployment/endpoint_metadata.json"))
                if meta.get("status") == "ACTIVE":
                    return {"status": ProjectStatus.DEPLOYED.value, "detail": "Model deployed to Vertex AI Endpoint"}
                elif meta.get("status") in ("DEPLOYING", "INITIALIZING"):
                    return {"status": ProjectStatus.DEPLOYING.value, "detail": meta.get("status_detail", "Deploying vLLM container to Vertex AI Endpoint")}
            except Exception:
                return {"status": ProjectStatus.DEPLOYED.value, "detail": "Model deployed to Vertex AI Endpoint"}

        # Check evaluation
        if self.file_exists(bucket_name, f"{p}evaluation/eval_results.json"):
            return {"status": ProjectStatus.EVALUATED.value, "detail": "3-tier evaluation complete"}

        # Check training completion
        if (self.file_exists(bucket_name, f"{p}training/final_adapter/adapter_model.safetensors") or
                self.file_exists(bucket_name, f"{p}training/final_adapter/adapter_config.json")):
            return {"status": ProjectStatus.TRAINING_COMPLETED.value, "detail": "Training completed, PEFT adapter ready"}

        # Check cost estimate
        if self.file_exists(bucket_name, f"{p}cost/cost_estimate.json"):
            return {"status": ProjectStatus.COST_ESTIMATED.value, "detail": "Hardware probe and cost scorecard ready"}

        # Check teacher inferences
        if self.file_exists(bucket_name, f"{p}data/teacher_inferences.jsonl"):
            return {"status": ProjectStatus.TEACHER_INFERENCE_DONE.value, "detail": "Teacher reasoning & completions extracted"}

        # Check split dataset
        if self.file_exists(bucket_name, f"{p}data/split_dataset.jsonl"):
            return {"status": ProjectStatus.DATASET_READY.value, "detail": "Dataset validated and split into train/val/test"}

        # Check master configuration
        if self.file_exists(bucket_name, f"{p}config.yaml"):
            return {"status": ProjectStatus.CONFIGURED.value, "detail": "Master configuration exists"}

        return {"status": ProjectStatus.UNINITIALIZED.value, "detail": "Project missing config.yaml"}

    def set_active_operation(self, bucket_name: str, project_id: str, operation: Optional[str]) -> None:
        p = f"{project_id}/status.json"
        data = {}
        if self.file_exists(bucket_name, p):
            try:
                data = json.loads(self.read_file(bucket_name, p))
            except Exception:
                data = {}
        data["active_operation"] = operation
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.write_file(bucket_name, p, json.dumps(data, indent=2))


storage_service = StorageService()
