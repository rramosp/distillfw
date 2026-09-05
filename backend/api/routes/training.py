"""Training management API routes."""

from fastapi import APIRouter, Query, HTTPException, Body
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from backend.services.training import training_service
from backend.core.config import settings

router = APIRouter(prefix="/training", tags=["Training"])


class StartTrainingRequest(BaseModel):
    dry_run: bool = False


@router.post("/{project_id}/start")
def start_training(
    project_id: str,
    req: StartTrainingRequest = Body(default_factory=StartTrainingRequest),
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    try:
        return training_service.launch_training(bucket, project_id, dry_run=req.dry_run)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{project_id}/metrics")
def get_training_metrics(
    project_id: str,
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    return training_service.get_metrics(bucket, project_id)


@router.get("/{project_id}/heartbeat")
def get_training_heartbeat(
    project_id: str,
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    return training_service.get_heartbeat(bucket, project_id)
