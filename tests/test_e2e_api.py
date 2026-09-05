"""End-to-end integration tests for FastAPI backend and distillation stages."""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
BUCKET = "distillfw-workspaces"
PROJECT = "distill-gemma-math-v1"


def test_healthz():
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json()["status"] == "HEALTHY"


def test_list_buckets():
    res = client.get("/api/workspaces/buckets")
    assert res.status_code == 200
    buckets = res.json()
    assert BUCKET in buckets


def test_list_projects():
    res = client.get(f"/api/workspaces/projects?bucket={BUCKET}")
    assert res.status_code == 200
    projects = res.json()
    assert PROJECT in projects


def test_dataset_ready_status():
    res = client.get(f"/api/workspaces/{PROJECT}/status?bucket={BUCKET}")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in [
        "DATASET_READY", "TEACHER_INFERENCE_DONE", "COST_ESTIMATED",
        "TRAINING_RUNNING", "TRAINING_COMPLETED", "EVALUATING", "EVALUATED", "DEPLOYED"
    ]


def test_get_config():
    res = client.get(f"/api/config/{PROJECT}?bucket={BUCKET}")
    assert res.status_code == 200
    cfg = res.json()
    assert cfg["project"]["id"] == PROJECT
    assert cfg["models"]["teacher"]["model_name"] == "gemini-2.5-pro"
    assert cfg["distillation"]["method"] == "cot_distillation"


def test_dataset_summary():
    res = client.get(f"/api/dataset/{PROJECT}/summary?bucket={BUCKET}")
    assert res.status_code == 200
    data = res.json()
    assert data["has_dataset"] is True
    assert data["counts"]["total"] == 100
    assert data["counts"]["train"] == 80
    assert data["counts"]["val"] == 10
    assert data["counts"]["test"] == 10


def test_cost_probe():
    res = client.post(f"/api/cost/{PROJECT}/probe?bucket={BUCKET}")
    assert res.status_code == 200
    probe = res.json()
    assert "hardware_probe" in probe
    assert "summary" in probe
    assert probe["summary"]["total_experiment_cost_usd"] > 0


def test_teacher_inference_and_status():
    res = client.post(f"/api/teacher/{PROJECT}/run?bucket={BUCKET}", json={"limit": 5})
    assert res.status_code == 200

    # Wait for completion or check status
    status_res = client.get(f"/api/teacher/{PROJECT}/status?bucket={BUCKET}")
    assert status_res.status_code == 200


def test_training_and_metrics():
    res = client.post(f"/api/training/{PROJECT}/start?bucket={BUCKET}", json={"dry_run": True})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["STARTED", "SUBMITTED"]

    metrics_res = client.get(f"/api/training/{PROJECT}/metrics?bucket={BUCKET}")
    assert metrics_res.status_code == 200


def test_evaluation():
    res = client.post(f"/api/evaluation/{PROJECT}/run?bucket={BUCKET}")
    assert res.status_code == 200

    eval_res = client.get(f"/api/evaluation/{PROJECT}/results?bucket={BUCKET}")
    if eval_res.status_code == 200:
        results = eval_res.json()
        assert "lexical_metrics" in results
        assert "operational_benchmarks" in results


def test_deployment_and_predict():
    res = client.post(f"/api/deployment/{PROJECT}/deploy?bucket={BUCKET}")
    assert res.status_code == 200
    meta = res.json()
    assert meta["status"] == "ACTIVE"

    pred_res = client.post(
        f"/api/deployment/{PROJECT}/predict?bucket={BUCKET}",
        json={"prompt": "What is 15 * 18?", "temperature": 0.2}
    )
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert pred_data["completion"] == "270"
    assert "latency_ms" in pred_data


def test_history_and_logs():
    h_res = client.get(f"/api/workspaces/{PROJECT}/history?bucket={BUCKET}")
    assert h_res.status_code == 200
    history = h_res.json()
    assert len(history) > 0

    logs_res = client.get(f"/api/logs?project_id={PROJECT}")
    assert logs_res.status_code == 200
    logs = logs_res.json()
    assert len(logs) > 0
