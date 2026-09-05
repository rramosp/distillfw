"""Cost probe API routes."""

from fastapi import APIRouter, Query, HTTPException
from typing import Dict, Any, Optional

from backend.services.cost_probe import cost_probe_service
from backend.core.config import settings

router = APIRouter(prefix="/cost", tags=["Cost & Probe"])


@router.post("/{project_id}/probe")
def run_cost_probe(
    project_id: str,
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    try:
        return cost_probe_service.calculate_probe(bucket, project_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{project_id}/estimate")
def get_cost_estimate(
    project_id: str,
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    est = cost_probe_service.get_estimate(bucket, project_id)
    if not est:
        raise HTTPException(status_code=404, detail="No cost estimate found. Run probe first.")
    return est
