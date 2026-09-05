"""Operations logs API routes."""

from fastapi import APIRouter, Query
from typing import List, Dict, Any, Optional

from backend.services.logger import operations_logger

router = APIRouter(prefix="/logs", tags=["Operations Logs"])


@router.get("")
def get_logs(
    project_id: Optional[str] = None,
    limit: int = Query(default=200, ge=1, le=1000)
) -> List[Dict[str, Any]]:
    return operations_logger.get_logs(project_id=project_id, limit=limit)


@router.post("/clear")
def clear_logs():
    operations_logger.clear()
    return {"status": "CLEARED"}
