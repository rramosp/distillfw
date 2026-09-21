"""Tests for configuration schemas, multi-task GCS tracking, and stateless resumption."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distillfw.config import DistillationConfig, FormatConfig, GCPConfig
from distillfw.gcp.storage import StorageBackend
from distillfw.state import StageName, StageStatus, TaskWorkspace


def test_config_yaml_roundtrip_and_splits() -> None:
    cfg = DistillationConfig(
        task_id="test-task-alpha",
        gcp=GCPConfig(project_id="proj-1", bucket_name="bucket-1"),
    )
    yaml_str = cfg.to_yaml()
    loaded = DistillationConfig.from_yaml(yaml_str)
    assert loaded.task_id == "test-task-alpha"
    assert loaded.task_uri == "gs://bucket-1/tasks/test-task-alpha"
    assert loaded.sha256() == cfg.sha256()

    with pytest.raises(ValueError, match="sum to 1.0"):
        FormatConfig(train_split_ratio=0.5, val_split_ratio=0.5, test_split_ratio=0.5)


def test_isolated_multi_task_workspaces_and_stateless_resumption(tmp_path: Path) -> None:
    storage = StorageBackend(local_root=tmp_path)
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(json.dumps({"prompt": "Hello"}) + "\n", encoding="utf-8")

    cfg1 = DistillationConfig(
        task_id="task-001",
        gcp=GCPConfig(project_id="p", bucket_name="b"),
    )
    cfg2 = DistillationConfig(
        task_id="task-002",
        gcp=GCPConfig(project_id="p", bucket_name="b"),
    )

    ws1 = TaskWorkspace.initialize(cfg1, prompts_file, storage=storage)
    ws2 = TaskWorkspace.initialize(cfg2, prompts_file, storage=storage)

    # Mark Stage 1 complete on Task 1 only
    ws1.mark_stage_completed(StageName.DATASET_GENERATOR, artifacts={"shard_count": 1})

    # Re-attach using ONLY the GCS task URI (zero local state)
    reattached_ws1 = TaskWorkspace(task_uri="gs://b/tasks/task-001", storage=storage)
    reattached_ws2 = TaskWorkspace(task_uri="gs://b/tasks/task-002", storage=storage)

    assert reattached_ws1.next_pending_stage() == StageName.DATASET_FORMATTER
    assert reattached_ws2.next_pending_stage() == StageName.DATASET_GENERATOR
    assert reattached_ws1.load_state().stages[StageName.DATASET_GENERATOR].status == StageStatus.COMPLETED
    assert reattached_ws2.load_state().stages[StageName.DATASET_GENERATOR].status == StageStatus.PENDING
