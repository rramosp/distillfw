"""Tests for training service, local execution worker, and trainer module imports."""

import os
import time
import pytest
from backend.services.storage import storage_service
from backend.services.training import training_service


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.core.config.settings.LOCAL_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr("backend.core.config.settings.STORAGE_MODE", "local")
    storage_service.use_gcs = False
    return tmp_path


def test_trainer_module_imports_without_torch():
    """Verify that trainer module can be imported even when PyTorch is not present."""
    import trainer
    from trainer.train import main as trainer_main
    from trainer.callbacks import GCSProgressCallback
    from trainer.distillation_loss import compute_seq_kd_loss
    assert callable(trainer_main)
    assert callable(compute_seq_kd_loss)


def test_local_training_execution(temp_workspace):
    """Verify that launching training in dry_run / local worker mode executes without error."""
    bucket = "distillfw-workspaces"
    project = "test-training-project"

    # Setup project with config
    config_path = "examples/sample_config.yaml" if os.path.exists("examples/sample_config.yaml") else "sample_config.yaml"
    with open(config_path, "r") as f:
        cfg = f.read()

    storage_service.write_file(bucket, f"{project}/config.yaml", cfg)
    storage_service.write_file(
        bucket,
        f"{project}/data/teacher_inferences.jsonl",
        '{"prompt": "Calculate 15 * 18", "completion": "270", "thinking": "Step by step", "split": "train"}\n'
    )

    res = training_service.launch_training(bucket, project, dry_run=True)
    assert res["status"] in ["STARTED", "SUBMITTED"]

    # Wait for local thread to finish simulation (20 steps * 0.2s = ~4s)
    max_wait = 15
    start = time.time()
    completed = False
    while time.time() - start < max_wait:
        history = storage_service.get_history(bucket, project)
        if any(h.get("action") == "TRAINING" for h in history):
            completed = True
            break
        time.sleep(0.5)

    assert completed is True, "Training did not complete within timeout"

    # Verify metrics and heartbeat exist
    assert storage_service.file_exists(bucket, f"{project}/training/metrics.jsonl")
    assert storage_service.file_exists(bucket, f"{project}/training/heartbeat.json")
    assert storage_service.file_exists(bucket, f"{project}/training/final_adapter/adapter_config.json")
    assert storage_service.file_exists(bucket, f"{project}/training/final_adapter/adapter_model.safetensors")

    # Verify metrics content
    metrics = training_service.get_metrics(bucket, project)
    assert len(metrics) > 0
    assert "train_loss" in metrics[-1]

    # Verify history recorded SUCCESS
    history = storage_service.get_history(bucket, project)
    success_entries = [h for h in history if h.get("action") == "TRAINING" and h.get("status") == "SUCCESS"]
    assert len(success_entries) > 0
