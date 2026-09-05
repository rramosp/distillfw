"""Evaluation management API routes."""

from fastapi import APIRouter, Query, HTTPException
from typing import Dict, Any, Optional

from backend.services.evaluation import evaluation_service
from backend.core.config import settings

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.post("/{project_id}/run")
def run_evaluation(
    project_id: str,
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    try:
        evaluation_service.trigger_async(bucket, project_id)
        return {"status": "STARTED", "project_id": project_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{project_id}/results")
def get_evaluation_results(
    project_id: str,
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    res = evaluation_service.get_results(bucket, project_id)
    if not res:
        raise HTTPException(status_code=404, detail="No evaluation results found.")
    return res
