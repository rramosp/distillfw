"""Vertex AI wrappers for single-node training jobs and model serving endpoints."""

from __future__ import annotations

import os
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

from distillfw.config import DeploymentConfig, GCPConfig, TrainingConfig
from distillfw.gcp.storage import StorageBackend

JOB_STATE_INT_MAP: dict[int, str] = {
    0: "JOB_STATE_UNSPECIFIED",
    1: "JOB_STATE_QUEUED",
    2: "JOB_STATE_PENDING",
    3: "JOB_STATE_RUNNING",
    4: "JOB_STATE_SUCCEEDED",
    5: "JOB_STATE_FAILED",
    6: "JOB_STATE_CANCELLING",
    7: "JOB_STATE_CANCELLED",
    8: "JOB_STATE_PAUSED",
    9: "JOB_STATE_EXPIRED",
    10: "JOB_STATE_UPDATING",
    11: "JOB_STATE_PARTIALLY_SUCCEEDED",
}


def normalize_vertex_job_state(raw_state: Any) -> str:
    """Normalize a Vertex AI JobState (IntEnum, int, or str) to canonical 'JOB_STATE_*' string."""
    if hasattr(raw_state, "name") and isinstance(raw_state.name, str):
        return raw_state.name
    if isinstance(raw_state, int):
        return JOB_STATE_INT_MAP.get(raw_state, f"JOB_STATE_{raw_state}")
    s = str(raw_state).strip()
    if s.isdigit():
        return JOB_STATE_INT_MAP.get(int(s), f"JOB_STATE_{s}")
    return s.split(".")[-1]


class VertexJobManager:
    """Manages single-node Vertex AI Custom Training Jobs."""

    def __init__(self, gcp_config: GCPConfig) -> None:
        self.gcp_config = gcp_config

    def _upload_package_to_gcs(self, task_uri: str) -> str:
        """Package local `distillfw` source tree into a `.tar.gz` and upload to `<task_uri>/00_inputs/`."""
        repo_root = Path(__file__).resolve().parents[3]
        package_gcs_uri = f"{task_uri.rstrip('/')}/00_inputs/distillfw_package.tar.gz"

        storage = StorageBackend(project_id=self.gcp_config.project_id)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_path = Path(tmp_dir) / "distillfw_package.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                for rel_name in ("pyproject.toml", "README.md", "src"):
                    candidate = repo_root / rel_name
                    if candidate.exists():
                        tar.add(
                            candidate,
                            arcname=f"distillfw_pkg/{rel_name}",
                            filter=lambda ti: None
                            if ("__pycache__" in ti.name or ti.name.endswith(".pyc"))
                            else ti,
                        )
            storage.upload_file(tar_path, package_gcs_uri)
        return package_gcs_uri

    @staticmethod
    def _resolve_hf_token() -> str | None:
        """Resolve Hugging Face token from environment or local HF cache for gated Gemma models."""
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            return token.strip()
        cache_token_path = Path.home() / ".cache" / "huggingface" / "token"
        if cache_token_path.exists():
            return cache_token_path.read_text(encoding="utf-8").strip()
        return None

    def submit_training_job(
        self,
        display_name: str,
        task_uri: str,
        training_config: TrainingConfig,
    ) -> dict[str, Any]:
        """Submit a single-node multi-GPU Vertex AI Custom Training Job.

        Uploads the local `distillfw` package archive to `<task_uri>/00_inputs/distillfw_package.tar.gz`,
        installs it inside the Vertex AI prebuilt PyTorch GPU container, and executes
        `python3 -m distillfw.cli run-stage <task_uri> --stage model_trainer --local-exec`.
        """
        from google.cloud import aiplatform

        staging_bucket = (
            self.gcp_config.staging_bucket or f"gs://{self.gcp_config.bucket_name}/staging"
        )
        aiplatform.init(
            project=self.gcp_config.project_id,
            location=self.gcp_config.location,
            staging_bucket=staging_bucket,
        )

        package_gcs_uri = self._upload_package_to_gcs(task_uri)
        hf_token = self._resolve_hf_token()
        env_vars = [{"name": "HF_TOKEN", "value": hf_token}] if hf_token else []

        bootstrap_script = (
            f"set -e && "
            f"python3 -c \"from google.cloud import storage; "
            f"client = storage.Client(); "
            f"b, p = '{package_gcs_uri}'.replace('gs://', '').split('/', 1); "
            f"client.bucket(b).blob(p).download_to_filename('/tmp/distillfw_package.tar.gz')\" && "
            f"(python3 -m pip uninstall -y torchvision torchaudio torch_xla torchdata torchtext torch-tensorrt || true) && "
            f"python3 -m pip install --no-cache-dir --upgrade 'numpy>=1.26.0,<2.0.0' 'torch>=2.5.0' /tmp/distillfw_package.tar.gz && "
            f"python3 -c \"import numpy, scipy, torch, transformers, accelerate, peft, trl; "
            f"print('NumPy:', numpy.__version__, 'PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'Transformers:', transformers.__version__); "
            f"assert transformers.utils.is_torch_available(), 'transformers reports torch unavailable'\" && "
            f"python3 -m distillfw.cli run-stage '{task_uri}' --stage model_trainer --local-exec"
        )

        worker_pool_specs = [
            {
                "machine_spec": {
                    "machine_type": training_config.vertex_machine_type,
                    "accelerator_type": training_config.vertex_accelerator_type,
                    "accelerator_count": training_config.vertex_accelerator_count,
                },
                "replica_count": 1,  # Strictly single-node per specification
                "container_spec": {
                    "image_uri": training_config.vertex_container_uri,
                    "command": ["bash", "-c"],
                    "args": [bootstrap_script],
                    "env": env_vars,
                },
            }
        ]

        job = aiplatform.CustomJob(
            display_name=display_name,
            worker_pool_specs=worker_pool_specs,
        )
        job.submit()
        norm_state = normalize_vertex_job_state(job.state)
        return {
            "job_resource_name": job.resource_name,
            "display_name": display_name,
            "state": norm_state,
        }

    def get_job_status(self, job_resource_name: str) -> dict[str, Any]:
        """Retrieve status of an existing Vertex AI Custom Job."""
        from google.cloud import aiplatform

        aiplatform.init(
            project=self.gcp_config.project_id,
            location=self.gcp_config.location,
        )
        job = aiplatform.CustomJob.get(resource_name=job_resource_name)
        norm_state = normalize_vertex_job_state(job.state)
        return {
            "job_resource_name": job.resource_name,
            "display_name": job.display_name,
            "state": norm_state,
            "error": str(job.error) if getattr(job, "error", None) else None,
        }

    def wait_for_job_completion(
        self,
        job_resource_name: str,
        poll_interval_seconds: float = 30.0,
        on_poll_callback: Any = None,
    ) -> dict[str, Any]:
        """Poll a Vertex AI Custom Job until it reaches a terminal state."""
        from distillfw.logging_utils import get_logger

        logger = get_logger("gcp.vertex")
        terminal_success = {"JOB_STATE_SUCCEEDED"}
        terminal_failure = {
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_CANCELLING",
            "JOB_STATE_EXPIRED",
        }

        while True:
            status_info = self.get_job_status(job_resource_name)
            normalized_state = normalize_vertex_job_state(status_info["state"])
            status_info["normalized_state"] = normalized_state

            if on_poll_callback is not None:
                on_poll_callback(status_info)

            if normalized_state in terminal_success:
                logger.info(
                    "Vertex AI Custom Job '%s' succeeded (state=%s).",
                    job_resource_name,
                    normalized_state,
                )
                return status_info

            if normalized_state in terminal_failure:
                error_detail = status_info.get("error") or f"Job terminated with state {normalized_state}"
                raise RuntimeError(
                    f"Vertex AI Custom Training Job '{job_resource_name}' failed "
                    f"(state={normalized_state}): {error_detail}"
                )

            logger.info(
                "Vertex AI Custom Job '%s' is still running (state=%s). Polling again in %.0fs...",
                job_resource_name,
                normalized_state,
                poll_interval_seconds,
            )
            time.sleep(poll_interval_seconds)


class VertexEndpointManager:
    """Manages Vertex AI Model Registry uploads and Endpoint deployments."""

    def __init__(self, gcp_config: GCPConfig) -> None:
        self.gcp_config = gcp_config

    def deploy_model(
        self,
        task_id: str,
        model_artifact_uri: str,
        deploy_config: DeploymentConfig,
    ) -> dict[str, Any]:
        """Upload distilled Gemma weights to Vertex AI Model Registry and deploy to Endpoint."""
        from google.cloud import aiplatform

        aiplatform.init(
            project=self.gcp_config.project_id,
            location=self.gcp_config.location,
        )

        display_name = deploy_config.endpoint_display_name or f"distillfw-{task_id}"

        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=model_artifact_uri,
            serving_container_image_uri=deploy_config.serving_container_uri,
            serving_container_args=deploy_config.vllm_args,
        )

        endpoint = aiplatform.Endpoint.create(display_name=display_name)
        model.deploy(
            endpoint=endpoint,
            machine_type=deploy_config.machine_type,
            accelerator_type=deploy_config.accelerator_type,
            accelerator_count=deploy_config.accelerator_count,
            min_replica_count=deploy_config.min_replica_count,
            max_replica_count=deploy_config.max_replica_count,
            traffic_percentage=deploy_config.traffic_percentage,
        )

        return {
            "model_resource_name": model.resource_name,
            "endpoint_resource_name": endpoint.resource_name,
            "endpoint_display_name": display_name,
            "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def predict(
        self,
        endpoint_resource_name: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Invoke a deployed Vertex AI Endpoint and record latency metrics."""
        from google.cloud import aiplatform

        aiplatform.init(
            project=self.gcp_config.project_id,
            location=self.gcp_config.location,
        )
        endpoint = aiplatform.Endpoint(endpoint_resource_name)
        t0 = time.perf_counter()
        response = endpoint.predict(
            instances=[
                {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            ]
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        prediction_text = (
            response.predictions[0]
            if isinstance(response.predictions[0], str)
            else response.predictions[0].get("text", str(response.predictions[0]))
        )
        return {
            "prediction": prediction_text,
            "latency_ms": round(latency_ms, 2),
            "endpoint_resource_name": endpoint_resource_name,
        }
