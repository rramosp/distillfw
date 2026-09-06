"""Unit tests for model inference, dual-endpoint comparative benchmarking, and dynamic model configuration."""

import json
import pytest
from backend.services.storage import storage_service
from backend.services.deployment import deployment_service
from backend.services.gcp_resources import GCPResourcesService


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.core.config.settings.LOCAL_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr("backend.core.config.settings.STORAGE_MODE", "local")
    storage_service.use_gcs = False
    return tmp_path


def test_deployment_and_predict_flow(temp_workspace):
    bucket = "distillfw-workspaces"
    project = "test-infer-project"

    # 1. Setup config with a custom student model
    config_yaml = """
models:
  teacher:
    model_name: "gemini-2.5-pro"
  student:
    model_name_or_path: "meta-llama/Llama-3.2-3B"
prompt:
  instructions: "Solve the math problem precisely."
  template: "{instructions}\\n\\nProblem:\\n{prompt}\\n\\nSolution:"
deployment:
  serving_framework: "vllm"
  machine_type: "g2-standard-4"
  accelerator_type: "NVIDIA_L4"
"""
    storage_service.write_file(bucket, f"{project}/config.yaml", config_yaml)

    # 2. Simulate completed model training with adapter artifacts
    storage_service.write_file(
        bucket,
        f"{project}/training/final_adapter/adapter_config.json",
        '{"base_model_name_or_path": "meta-llama/Llama-3.2-3B", "r": 16}'
    )
    storage_service.write_file(
        bucket,
        f"{project}/training/final_adapter/adapter_model.safetensors",
        "safetensors_bytes"
    )

    # 3. Deploy endpoints synchronously
    meta = deployment_service.deploy_endpoint(bucket, project, sync=True)
    assert meta["status"] == "ACTIVE"
    assert "endpoint_base" in meta
    assert "endpoint_distilled" in meta
    assert meta["base_model"] == "meta-llama/Llama-3.2-3B"
    assert meta["endpoint_base"]["model"] == "meta-llama/Llama-3.2-3B"
    assert "meta-llama/Llama-3.2-3B" in meta["endpoint_distilled"]["model"]

    # 4. Predict using the deployed endpoint
    res = deployment_service.predict(
        bucket_name=bucket,
        project_id=project,
        prompt="Calculate 25 * 14",
        temperature=0.2
    )

    assert "student_before" in res
    assert "teacher" in res
    assert "student_after" in res

    # Verify model propagation
    assert "meta-llama/Llama-3.2-3B" in res["student_before"]["model"]
    assert "meta-llama/Llama-3.2-3B" in res["student_after"]["model"]
    assert res["teacher"]["model"] == "gemini-2.5-pro"

    # Verify latencies are genuine floats > 0
    assert isinstance(res["latency_ms"], (int, float))
    assert isinstance(res["student_before"]["latency_ms"], (int, float))
    assert isinstance(res["teacher"]["latency_ms"], (int, float))

    # 5. Check GCP resources list includes Model Registry entries
    gcp_service = GCPResourcesService()
    resources = gcp_service.get_workspace_resources(bucket, project)
    resource_ids = [r["id"] for r in resources.get("resources", [])]
    assert "vertex_endpoint_base" in resource_ids
    assert "vertex_endpoint_distilled" in resource_ids
    assert "vertex_model_base" in resource_ids
    assert "vertex_model_distilled" in resource_ids

    # 6. Clear deployment
    clear_res = deployment_service.clear(bucket, project)
    assert clear_res["status"] == "CLEARED"
    assert deployment_service.get_metadata(bucket, project) is None
