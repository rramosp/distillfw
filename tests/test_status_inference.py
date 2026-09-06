"""Tests for deterministic project status inference."""

import os
import shutil
import pytest
from backend.services.storage import storage_service
from backend.core.models import ProjectStatus


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.core.config.settings.LOCAL_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr("backend.core.config.settings.STORAGE_MODE", "local")
    storage_service.use_gcs = False
    return tmp_path


def test_uninitialized_status(temp_workspace):
    bucket = "test-bucket"
    project = "test-proj"
    res = storage_service.infer_status(bucket, project)
    assert res["status"] == ProjectStatus.UNINITIALIZED.value


def test_configured_status(temp_workspace):
    bucket = "test-bucket"
    project = "test-proj"
    storage_service.write_file(bucket, f"{project}/config.yaml", "project:\n  id: test-proj\n")
    res = storage_service.infer_status(bucket, project)
    assert res["status"] == ProjectStatus.CONFIGURED.value


def test_dataset_ready_status(temp_workspace):
    bucket = "test-bucket"
    project = "test-proj"
    storage_service.write_file(bucket, f"{project}/config.yaml", "project:\n  id: test-proj\n")
    storage_service.write_file(bucket, f"{project}/data/split_dataset.jsonl", '{"prompt": "1+1", "split": "train"}\n')
    res = storage_service.infer_status(bucket, project)
    assert res["status"] == ProjectStatus.DATASET_READY.value


def test_teacher_inference_done_status(temp_workspace):
    bucket = "test-bucket"
    project = "test-proj"
    storage_service.write_file(bucket, f"{project}/config.yaml", "project:\n  id: test-proj\n")
    storage_service.write_file(bucket, f"{project}/data/split_dataset.jsonl", '{"prompt": "1+1", "split": "train"}\n')
    storage_service.write_file(bucket, f"{project}/data/teacher_inferences.jsonl", '{"prompt": "1+1", "teacher_response": "2"}\n')
    res = storage_service.infer_status(bucket, project)
    assert res["status"] == ProjectStatus.TEACHER_INFERENCE_DONE.value


def test_cost_estimated_status(temp_workspace):
    bucket = "test-bucket"
    project = "test-proj"
    storage_service.write_file(bucket, f"{project}/config.yaml", "project:\n  id: test-proj\n")
    storage_service.write_file(bucket, f"{project}/data/split_dataset.jsonl", '{"prompt": "1+1", "split": "train"}\n')
    storage_service.write_file(bucket, f"{project}/data/teacher_inferences.jsonl", '{"prompt": "1+1", "teacher_response": "2"}\n')
    storage_service.write_file(bucket, f"{project}/cost/cost_estimate.json", '{"total_experiment_cost_usd": 12.5}\n')
    res = storage_service.infer_status(bucket, project)
    assert res["status"] == ProjectStatus.COST_ESTIMATED.value


def test_training_completed_status(temp_workspace):
    bucket = "test-bucket"
    project = "test-proj"
    storage_service.write_file(bucket, f"{project}/config.yaml", "project:\n  id: test-proj\n")
    storage_service.write_file(bucket, f"{project}/data/split_dataset.jsonl", '{"prompt": "1+1", "split": "train"}\n')
    storage_service.write_file(bucket, f"{project}/data/teacher_inferences.jsonl", '{"prompt": "1+1", "teacher_response": "2"}\n')
    storage_service.write_file(bucket, f"{project}/cost/cost_estimate.json", '{"total_experiment_cost_usd": 12.5}\n')
    storage_service.write_file(bucket, f"{project}/training/final_adapter/adapter_model.safetensors", "BIN_DATA")
    res = storage_service.infer_status(bucket, project)
    assert res["status"] == ProjectStatus.TRAINING_COMPLETED.value


def test_evaluated_status(temp_workspace):
    bucket = "test-bucket"
    project = "test-proj"
    storage_service.write_file(bucket, f"{project}/config.yaml", "project:\n  id: test-proj\n")
    storage_service.write_file(bucket, f"{project}/data/split_dataset.jsonl", '{"prompt": "1+1", "split": "train"}\n')
    storage_service.write_file(bucket, f"{project}/data/teacher_inferences.jsonl", '{"prompt": "1+1", "teacher_response": "2"}\n')
    storage_service.write_file(bucket, f"{project}/cost/cost_estimate.json", '{"total_experiment_cost_usd": 12.5}\n')
    storage_service.write_file(bucket, f"{project}/training/final_adapter/adapter_model.safetensors", "BIN_DATA")
    storage_service.write_file(bucket, f"{project}/evaluation/eval_results.json", '{"metrics": {}}\n')
    res = storage_service.infer_status(bucket, project)
    assert res["status"] == ProjectStatus.EVALUATED.value


def test_deployed_status(temp_workspace):
    bucket = "test-bucket"
    project = "test-proj"
    storage_service.write_file(bucket, f"{project}/config.yaml", "project:\n  id: test-proj\n")
    storage_service.write_file(bucket, f"{project}/data/split_dataset.jsonl", '{"prompt": "1+1", "split": "train"}\n')
    storage_service.write_file(bucket, f"{project}/data/teacher_inferences.jsonl", '{"prompt": "1+1", "teacher_response": "2"}\n')
    storage_service.write_file(bucket, f"{project}/cost/cost_estimate.json", '{"total_experiment_cost_usd": 12.5}\n')
    storage_service.write_file(bucket, f"{project}/training/final_adapter/adapter_model.safetensors", "BIN_DATA")
    storage_service.write_file(bucket, f"{project}/evaluation/eval_results.json", '{"metrics": {}}\n')
    storage_service.write_file(bucket, f"{project}/deployment/endpoint_metadata.json", '{"status": "ACTIVE"}\n')
    res = storage_service.infer_status(bucket, project)
    assert res["status"] == ProjectStatus.DEPLOYED.value


def test_deploying_status(temp_workspace):
    bucket = "test-bucket"
    project = "test-proj-deploying"
    storage_service.write_file(bucket, f"{project}/config.yaml", "project:\n  id: test-proj-deploying\n")
    storage_service.write_file(bucket, f"{project}/data/split_dataset.jsonl", '{"prompt": "1+1", "split": "train"}\n')
    storage_service.write_file(bucket, f"{project}/data/teacher_inferences.jsonl", '{"prompt": "1+1", "teacher_response": "2"}\n')
    storage_service.write_file(bucket, f"{project}/cost/cost_estimate.json", '{"total_experiment_cost_usd": 12.5}\n')
    storage_service.write_file(bucket, f"{project}/training/final_adapter/adapter_model.safetensors", "BIN_DATA")
    storage_service.write_file(bucket, f"{project}/evaluation/eval_results.json", '{"metrics": {}}\n')
    storage_service.write_file(bucket, f"{project}/deployment/endpoint_metadata.json", '{"status": "DEPLOYING", "progress_pct": 40}\n')
    res = storage_service.infer_status(bucket, project)
    assert res["status"] == ProjectStatus.DEPLOYING.value

