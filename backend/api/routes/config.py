"""Configuration management API routes."""

from fastapi import APIRouter, Query, HTTPException, Body
from typing import Dict, Any, Optional
import yaml

from backend.services.storage import storage_service
from backend.core.config import settings
from backend.core.models import MasterConfig
from backend.services.logger import operations_logger

router = APIRouter(prefix="/config", tags=["Configuration"])


@router.get("/{project_id}")
def get_config(
    project_id: str,
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    path = f"{project_id}/config.yaml"
    if storage_service.file_exists(bucket, path):
        try:
            raw = storage_service.read_file(bucket, path)
            data = yaml.safe_load(raw)
            # Validate with MasterConfig to ensure schema consistency
            validated = MasterConfig(**(data or {}))
            return validated.model_dump()
        except Exception as e:
            operations_logger.log(f"Error parsing existing config: {e}", level="WARNING", source="CONFIG", project_id=project_id)

    # Return default template initialized for this project
    default_cfg = MasterConfig()
    default_cfg.project.id = project_id
    default_cfg.project.gcs_workspace = f"gs://{bucket}/{project_id}"
    return default_cfg.model_dump()


@router.post("/{project_id}")
def save_config(
    project_id: str,
    payload: Dict[str, Any] = Body(...),
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    try:
        # Validate against schema
        validated = MasterConfig(**payload)
        # Ensure project id and workspace are aligned
        validated.project.id = project_id
        if not validated.project.gcs_workspace or validated.project.gcs_workspace == "":
            validated.project.gcs_workspace = f"gs://{bucket}/{project_id}"

        # Convert to YAML
        yaml_content = yaml.dump(validated.model_dump(), sort_keys=False)
        path = f"{project_id}/config.yaml"
        storage_service.write_file(bucket, path, yaml_content)

        storage_service.record_history(
            bucket, project_id, "CONFIG_UPDATE", "SUCCESS",
            {"path": path},
            "Saved and validated master configuration config.yaml"
        )
        operations_logger.log(f"Updated configuration for '{project_id}'", level="SUCCESS", source="CONFIG", project_id=project_id)

        return {"success": True, "config": validated.model_dump()}
    except Exception as e:
        operations_logger.log(f"Failed to save config: {e}", level="ERROR", source="CONFIG", project_id=project_id)
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
