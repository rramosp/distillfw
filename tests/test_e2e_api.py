"""End-to-end integration tests for FastAPI backend and distillation stages."""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
BUCKET = "distillfw-workspaces"
PROJECT = "distill-gemma-math-v1"


@pytest.fixture(autouse=True, scope="module")
def setup_workspace():
    from backend.services.storage import storage_service
    from backend.services.dataset import dataset_service

    with open("examples/sample_config.yaml", "r") as f:
        cfg_text = f.read()
    with open("examples/sample_dataset.jsonl", "r") as f:
        ds_text = f.read()

    storage_service.write_file(BUCKET, f"{PROJECT}/config.yaml", cfg_text)
    dataset_service.ingest_and_split(BUCKET, PROJECT, ds_text)
    yield


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
    # 3-Model Comparison verification
    assert "student_before" in pred_data
    assert "teacher" in pred_data
    assert "student_after" in pred_data
    assert "270" in pred_data["student_before"]["completion"]
    assert pred_data["teacher"]["completion"] == "270"
    assert "thinking" in pred_data["teacher"]
    assert pred_data["student_after"]["completion"] == "270"
    assert pred_data["student_after"]["serving_framework"] == "vllm"


def test_teacher_retries_diagnostics():
    retries_res = client.get(f"/api/teacher/{PROJECT}/retries?bucket={BUCKET}")
    assert retries_res.status_code == 200
    r_data = retries_res.json()
    assert "retries_count" in r_data
    assert "error_types" in r_data
    assert "errors_encountered" in r_data

    status_res = client.get(f"/api/teacher/{PROJECT}/status?bucket={BUCKET}")
    assert status_res.status_code == 200
    s_data = status_res.json()
    assert "retries_count" in s_data
    assert "error_types" in s_data


def test_stop_and_clear_lifecycle():
    LIFECYCLE_PROJECT = "distill-test-lifecycle"
    from backend.services.storage import storage_service
    from backend.services.dataset import dataset_service

    with open("examples/sample_config.yaml", "r") as f:
        cfg_text = f.read()
    with open("examples/sample_dataset.jsonl", "r") as f:
        ds_text = f.read()

    storage_service.write_file(BUCKET, f"{LIFECYCLE_PROJECT}/config.yaml", cfg_text)
    dataset_service.ingest_and_split(BUCKET, LIFECYCLE_PROJECT, ds_text)

    # Test stop endpoints
    assert client.post(f"/api/teacher/{LIFECYCLE_PROJECT}/stop?bucket={BUCKET}").status_code == 200
    assert client.post(f"/api/cost/{LIFECYCLE_PROJECT}/stop?bucket={BUCKET}").status_code == 200
    assert client.post(f"/api/training/{LIFECYCLE_PROJECT}/stop?bucket={BUCKET}").status_code == 200
    assert client.post(f"/api/evaluation/{LIFECYCLE_PROJECT}/stop?bucket={BUCKET}").status_code == 200
    assert client.post(f"/api/deployment/{LIFECYCLE_PROJECT}/stop?bucket={BUCKET}").status_code == 200

    # Test clear endpoints (clearing artifacts to start over)
    assert client.post(f"/api/deployment/{LIFECYCLE_PROJECT}/clear?bucket={BUCKET}").status_code == 200
    assert client.post(f"/api/evaluation/{LIFECYCLE_PROJECT}/clear?bucket={BUCKET}").status_code == 200
    assert client.post(f"/api/training/{LIFECYCLE_PROJECT}/clear?bucket={BUCKET}").status_code == 200
    assert client.post(f"/api/cost/{LIFECYCLE_PROJECT}/clear?bucket={BUCKET}").status_code == 200
    assert client.post(f"/api/teacher/{LIFECYCLE_PROJECT}/clear?bucket={BUCKET}").status_code == 200
    assert client.post(f"/api/dataset/{LIFECYCLE_PROJECT}/clear?bucket={BUCKET}").status_code == 200


def test_history_and_logs():
    h_res = client.get(f"/api/workspaces/{PROJECT}/history?bucket={BUCKET}")
    assert h_res.status_code == 200
    history = h_res.json()
    assert len(history) > 0

    logs_res = client.get(f"/api/logs?project_id={PROJECT}")
    assert logs_res.status_code == 200
    logs = logs_res.json()
    assert len(logs) > 0


def test_workspace_gcp_resources():
    res = client.get(f"/api/workspaces/{PROJECT}/resources?bucket={BUCKET}")
    assert res.status_code == 200
    data = res.json()

    assert data["project_id"] == PROJECT
    assert data["bucket"] == BUCKET
    assert "gcp_project_id" in data
    assert "region" in data
    assert "summary" in data
    assert data["summary"]["total_resources"] >= 10

    resources = data["resources"]
    assert len(resources) >= 10

    resource_services = [r["service"] for r in resources]
    assert "Cloud Storage" in resource_services
    assert "Vertex AI Training" in resource_services
    assert "Vertex AI Prediction" in resource_services
    assert "Vertex AI Model Registry" in resource_services
    assert "Vertex AI Gemini API" in resource_services
    assert "Artifact Registry" in resource_services
    assert "Cloud IAM" in resource_services
    assert "Cloud Run" in resource_services
    assert "Cloud Logging" in resource_services

    for r in resources:
        assert "id" in r
        assert "name" in r
        assert "service" in r
        assert "status" in r
        assert "status_detail" in r
        assert "console_url" in r
        assert r["console_url"].startswith("https://console.cloud.google.com/")
        assert f"project={data['gcp_project_id']}" in r["console_url"]


