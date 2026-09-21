"""Vertex AI wrappers for single-node training jobs and model serving endpoints."""

from __future__ import annotations

import time
from typing import Any

from distillfw.config import DeploymentConfig, GCPConfig, TrainingConfig


class VertexJobManager:
    """Manages single-node Vertex AI Custom Training Jobs."""

    def __init__(self, gcp_config: GCPConfig) -> None:
        self.gcp_config = gcp_config

    def submit_training_job(
        self,
        display_name: str,
        task_uri: str,
        training_config: TrainingConfig,
    ) -> dict[str, Any]:
        """Submit a single-node multi-GPU Vertex AI Custom Training Job.

        The job receives only `--task-uri gs://.../<task_id>` and resumes
        execution using the self-contained GCS task state.
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
                    "command": ["distillfw"],
                    "args": ["run-stage", task_uri, "--stage", "model_trainer", "--local-exec"],
                },
            }
        ]

        job = aiplatform.CustomJob(
            display_name=display_name,
            worker_pool_specs=worker_pool_specs,
        )
        job.submit()
        return {
            "job_resource_name": job.resource_name,
            "display_name": display_name,
            "state": str(job.state),
        }

    def get_job_status(self, job_resource_name: str) -> dict[str, Any]:
        """Retrieve status of an existing Vertex AI Custom Job."""
        from google.cloud import aiplatform

        aiplatform.init(
            project=self.gcp_config.project_id,
            location=self.gcp_config.location,
        )
        job = aiplatform.CustomJob.get(resource_name=job_resource_name)
        return {
            "job_resource_name": job.resource_name,
            "display_name": job.display_name,
            "state": str(job.state),
            "error": str(job.error) if getattr(job, "error", None) else None,
        }


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
