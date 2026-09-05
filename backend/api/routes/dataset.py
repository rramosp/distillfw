"""Dataset management API routes."""

from fastapi import APIRouter, Query, HTTPException, UploadFile, File, Body
from typing import Dict, Any, Optional
from pydantic import BaseModel

from backend.services.storage import storage_service
from backend.services.dataset import dataset_service
from backend.core.config import settings

router = APIRouter(prefix="/dataset", tags=["Dataset"])


class SplitRequest(BaseModel):
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42


class RawDatasetUpload(BaseModel):
    content: str


@router.post("/{project_id}/upload")
def upload_dataset(
    project_id: str,
    payload: RawDatasetUpload,
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    if not payload.content or not payload.content.strip():
        raise HTTPException(status_code=400, detail="Dataset content is empty")

    res = dataset_service.ingest_and_split(
        bucket_name=bucket,
        project_id=project_id,
        raw_jsonl_content=payload.content
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail={"errors": res.get("errors")})
    return res


@router.post("/{project_id}/split")
def split_existing_dataset(
    project_id: str,
    req: SplitRequest,
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    input_path = f"{project_id}/data/input_dataset.jsonl"
    if not storage_service.file_exists(bucket, input_path):
        raise HTTPException(status_code=404, detail="No input dataset found. Please upload input_dataset.jsonl first.")

    raw = storage_service.read_file(bucket, input_path)
    res = dataset_service.ingest_and_split(
        bucket_name=bucket,
        project_id=project_id,
        raw_jsonl_content=raw,
        train_ratio=req.train_ratio,
        val_ratio=req.val_ratio,
        test_ratio=req.test_ratio,
        random_seed=req.random_seed
    )
    return res


@router.get("/{project_id}/summary")
def get_dataset_summary(
    project_id: str,
    bucket: str = Query(default_factory=lambda: settings.DEFAULT_BUCKET)
):
    return dataset_service.get_summary(bucket, project_id)
