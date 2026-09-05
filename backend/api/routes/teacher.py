"""Teacher inference API routes."""

from fastapi import APIRouter, Query, HTTPException, Body
from typing import Dict, Any, Optional
from pydantic import BaseModel

from backend.services.teacher import teacher_service
from backend.core.config import settings

router = APIRouter(prefix="/teacher", tags=["Teacher Inference"])


class RunInferenceRequest(BaseModel):
    limit: Optional[int] = None


@router.post("/{project_id}/run")
def trigger_teacher_inference(
    project_id: str,
    req: RunInferenceRequest = Body(default_factory=RunInferenceRequest),
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    teacher_service.trigger_async(bucket, project_id, limit=req.limit)
    return {"status": "STARTED", "project_id": project_id, "limit": req.limit}


@router.get("/{project_id}/status")
def get_teacher_status(
    project_id: str,
    limit: int = Query(default=10),
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    return teacher_service.get_inferences(bucket, project_id, limit=limit)
