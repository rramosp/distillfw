"""Tests for cost estimation and hardware probe service."""

import os
import yaml
import pytest
from backend.services.storage import storage_service
from backend.services.cost_probe import cost_probe_service


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.core.config.settings.LOCAL_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr("backend.core.config.settings.STORAGE_MODE", "local")
    storage_service.use_gcs = False
    return tmp_path


def test_cost_probe_calculation(temp_workspace):
    bucket = "distillfw-workspaces"
    project = "test-cost-proj"

    # Setup config and dataset
    with open("sample_config.yaml", "r") as f:
        cfg = f.read()
    storage_service.write_file(bucket, f"{project}/config.yaml", cfg)
    storage_service.write_file(
        bucket,
        f"{project}/data/split_dataset.jsonl",
        '{"prompt": "Calculate 15 * 18", "split": "train"}\n' * 50 +
        '{"prompt": "Calculate 2 + 2", "split": "test"}\n' * 10
    )

    probe = cost_probe_service.calculate_probe(bucket, project)
    assert "teacher_inference" in probe
    assert "hardware_probe" in probe
    assert "summary" in probe
    assert probe["summary"]["total_experiment_cost_usd"] > 0
    assert probe["hardware_probe"]["peak_vram_gb"] < probe["hardware_probe"]["vram_limit_gb"]
    assert probe["summary"]["recommended"] is True
