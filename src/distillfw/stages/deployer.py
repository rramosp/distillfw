"""Stage 5: Model deployment to Vertex AI Endpoints."""

from __future__ import annotations

from typing import Any, Callable

from distillfw.gcp.vertex import VertexEndpointManager
from distillfw.state import StageName, TaskWorkspace


class ModelDeployer:
    """Stage 5: Registers distilled Gemma weights and deploys to a Vertex AI Endpoint.

    Stateless & Resumable:
    - Reads exported model weights strictly from `<task_uri>/05_exported_model/`.
    - Writes live endpoint metadata to `<task_uri>/06_deployment/endpoint_info.json`.
    """

    def __init__(
        self,
        workspace: TaskWorkspace,
        deploy_fn: Callable[[str, str, Any], dict[str, Any]] | None = None,
    ) -> None:
        self.workspace = workspace
        self.deploy_fn = deploy_fn

    def run(self) -> dict[str, Any]:
        """Execute Stage 5 deployment to Vertex AI Endpoints."""
        config = self.workspace.load_config()
        self.workspace.mark_stage_running(StageName.MODEL_DEPLOYER)

        try:
            model_artifact_uri = self.workspace.exported_model_dir_uri
            if self.deploy_fn is not None:
                endpoint_info = self.deploy_fn(
                    config.task_id, model_artifact_uri, config.deployment
                )
            else:
                endpoint_mgr = VertexEndpointManager(config.gcp)
                endpoint_info = endpoint_mgr.deploy_model(
                    task_id=config.task_id,
                    model_artifact_uri=model_artifact_uri,
                    deploy_config=config.deployment,
                )

            manifest_uri = f"{self.workspace.deployment_dir_uri}/endpoint_info.json"
            self.workspace.storage.write_json(manifest_uri, endpoint_info)

            artifacts = {
                "endpoint_info_uri": manifest_uri,
                "endpoint_info": endpoint_info,
            }
            self.workspace.mark_stage_completed(StageName.MODEL_DEPLOYER, artifacts=artifacts)
            return artifacts
        except Exception as exc:
            self.workspace.mark_stage_failed(StageName.MODEL_DEPLOYER, str(exc))
            raise
