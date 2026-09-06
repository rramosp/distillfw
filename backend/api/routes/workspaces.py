"""Workspace and bucket management API routes."""

from fastapi import APIRouter, Query, HTTPException, Body
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from backend.services.storage import storage_service
from backend.services.gcp_resources import gcp_resources_service
from backend.core.config import settings
from backend.core.models import ProjectStatus

router = APIRouter(prefix="/workspaces", tags=["Workspaces"])


class CreateProjectRequest(BaseModel):
    project_id: str
    bucket: Optional[str] = None
    description: Optional[str] = None


@router.get("/buckets", response_model=List[str])
def list_buckets():
    return storage_service.list_buckets()


@router.get("/projects", response_model=List[str])
def list_projects(bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)):
    return storage_service.list_projects(bucket)


@router.post("/projects")
def create_project(req: CreateProjectRequest):
    bucket = req.bucket or settings.DEFAULT_BUCKET
    if not req.project_id or not req.project_id.strip():
        raise HTTPException(status_code=400, detail="Project ID cannot be empty")
    return storage_service.create_project(bucket, req.project_id.strip(), req.description)


@router.get("/{project_id}/status")
def get_project_status(
    project_id: str,
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    return storage_service.infer_status(bucket, project_id)


@router.get("/{project_id}/history")
def get_project_history(
    project_id: str,
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    return storage_service.get_history(bucket, project_id)


@router.get("/{project_id}/resources")
def get_project_resources(
    project_id: str,
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    return gcp_resources_service.get_workspace_resources(bucket, project_id)

