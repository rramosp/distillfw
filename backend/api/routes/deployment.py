"""Deployment management API routes."""

from fastapi import APIRouter, Query, HTTPException, Body
from typing import Dict, Any, Optional
from pydantic import BaseModel

from backend.services.deployment import deployment_service
from backend.core.config import settings

router = APIRouter(prefix="/deployment", tags=["Deployment"])


class PredictRequest(BaseModel):
    prompt: str
    temperature: float = 0.2
    max_tokens: int = 256


@router.post("/{project_id}/deploy")
def deploy_model(
    project_id: str,
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    try:
        return deployment_service.deploy_endpoint(bucket, project_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{project_id}/status")
def get_deployment_status(
    project_id: str,
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    meta = deployment_service.get_metadata(bucket, project_id)
    if not meta:
        raise HTTPException(status_code=404, detail="No deployment metadata found.")
    return meta


@router.post("/{project_id}/predict")
def predict(
    project_id: str,
    req: PredictRequest,
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    try:
        return deployment_service.predict(
            bucket_name=bucket,
            project_id=project_id,
            prompt=req.prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
