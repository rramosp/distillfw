"""Tests for configuration schemas, multi-task GCS tracking, and stateless resumption."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distillfw.config import DistillationConfig, FormatConfig, GCPConfig, LocalConfig
from distillfw.gcp.storage import StorageBackend
from distillfw.state import StageName, StageStatus, TaskWorkspace


def test_config_yaml_roundtrip_and_splits(tmp_path: Path) -> None:
    cfg = DistillationConfig(
        task_id="test-task-alpha",
        gcp=GCPConfig(project_id="proj-1", bucket_name="bucket-1"),
    )
    assert cfg.training.execution_mode == "vertex_custom_job"
    yaml_str = cfg.to_yaml()
    loaded = DistillationConfig.from_yaml(yaml_str)
    assert loaded.task_id == "test-task-alpha"
    assert loaded.task_uri == "gs://bucket-1/tasks/test-task-alpha"
    assert loaded.local is None
    assert loaded.sha256() == cfg.sha256()

    # Local config roundtrip
    local_cfg = DistillationConfig(
        task_id="test-local-task",
        local=LocalConfig(storage_root=str(tmp_path / "local_tasks")),
    )
    assert local_cfg.task_uri == f"{tmp_path / 'local_tasks'}/test-local-task"
    local_yaml = local_cfg.to_yaml()
    loaded_local = DistillationConfig.from_yaml(local_yaml)
    assert loaded_local.gcp is None
    assert loaded_local.local is not None
    assert loaded_local.local.storage_root == str(tmp_path / "local_tasks")
    assert loaded_local.task_uri == local_cfg.task_uri

    # Specifying neither gcp nor local must fail validation
    with pytest.raises(ValueError, match="Exactly one of 'gcp' or 'local'"):
        DistillationConfig(task_id="invalid-neither")

    # Specifying both gcp and local must fail validation
    with pytest.raises(ValueError, match="Exactly one of 'gcp' or 'local'"):
        DistillationConfig(
            task_id="invalid-both",
            gcp=GCPConfig(project_id="proj-1", bucket_name="bucket-1"),
            local=LocalConfig(storage_root="/tmp/tasks"),
        )

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


def test_bundled_examples_configs_and_datasets_valid() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    for example_name in ("support_ticket_classification", "query_expansion"):
        example_dir = repo_root / "examples" / example_name
        cfg = DistillationConfig.from_yaml(example_dir / "config.yaml")
        assert cfg.teacher.model_id == "gemini-3.5-flash"
        assert cfg.teacher.location == "global"
        lines = (example_dir / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        assert 100 <= len(rows) <= 200
        assert all("prompt" in r and len(r["prompt"]) > 20 for r in rows)


def test_init_preflight_logprobs_compatibility_and_probe(tmp_path: Path) -> None:
    from distillfw.config import TeacherConfig, TrainingAlgorithm, TrainingConfig
    from distillfw.state import TaskInitializationError

    storage = StorageBackend(local_root=tmp_path)
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(json.dumps({"prompt": "Hello"}) + "\n", encoding="utf-8")

    # 1. Algorithm does not need logprobs (sft_seqkd) but response_logprobs=True -> error
    cfg_bad_non_logprob = DistillationConfig(
        task_id="bad-non-logprob",
        gcp=GCPConfig(project_id="p", bucket_name="b"),
        teacher=TeacherConfig(response_logprobs=True, logprobs_top_k=5),
        training=TrainingConfig(algorithm=TrainingAlgorithm.SFT_SEQKD),
    )
    with pytest.raises(TaskInitializationError, match="does not use teacher logprobs"):
        TaskWorkspace.initialize(cfg_bad_non_logprob, prompts_file, storage=storage)

    # 2. Algorithm needs logprobs (skew_kl) but response_logprobs=False -> error
    cfg_missing_flag = DistillationConfig(
        task_id="missing-flag",
        gcp=GCPConfig(project_id="p", bucket_name="b"),
        teacher=TeacherConfig(response_logprobs=False),
        training=TrainingConfig(algorithm=TrainingAlgorithm.SKEW_KL),
    )
    with pytest.raises(TaskInitializationError, match="requires teacher logprobs"):
        TaskWorkspace.initialize(cfg_missing_flag, prompts_file, storage=storage)

    # 3. Algorithm needs logprobs (skew_kl) with response_logprobs=True but no logprobs_top_k -> error
    cfg_missing_topk = DistillationConfig(
        task_id="missing-topk",
        gcp=GCPConfig(project_id="p", bucket_name="b"),
        teacher=TeacherConfig(response_logprobs=True, logprobs_top_k=None),
        training=TrainingConfig(algorithm=TrainingAlgorithm.SKEW_KL),
    )
    with pytest.raises(TaskInitializationError, match="requires teacher.logprobs_top_k"):
        TaskWorkspace.initialize(cfg_missing_topk, prompts_file, storage=storage)

    # 4. Probe fails when teacher returns empty logprobs or fewer candidates than top_k
    cfg_valid_logprob = DistillationConfig(
        task_id="valid-logprob",
        gcp=GCPConfig(project_id="p", bucket_name="b"),
        teacher=TeacherConfig(response_logprobs=True, logprobs_top_k=3),
        training=TrainingConfig(algorithm=TrainingAlgorithm.SKEW_KL),
    )
    with pytest.raises(TaskInitializationError, match="did not return any token logprobs"):
        TaskWorkspace.initialize(
            cfg_valid_logprob,
            prompts_file,
            storage=storage,
            teacher_callable=lambda _prompt, _cfg: {"response": "OK", "topk_logprobs": []},
        )
    assert not storage.exists(f"{cfg_valid_logprob.task_uri}/task_state.json")

    with pytest.raises(TaskInitializationError, match="returned only 2 logprob candidates"):
        TaskWorkspace.initialize(
            cfg_valid_logprob,
            prompts_file,
            storage=storage,
            teacher_callable=lambda _prompt, _cfg: {
                "response": "OK",
                "topk_logprobs": [
                    {
                        "top_candidates": [
                            {"token": "A", "log_probability": -0.1},
                            {"token": "B", "log_probability": -1.2},
                        ]
                    }
                ],
            },
        )
    assert not storage.exists(f"{cfg_valid_logprob.task_uri}/task_state.json")

    # 5. Probe succeeds when teacher returns at least logprobs_top_k entries per step
    ws = TaskWorkspace.initialize(
        cfg_valid_logprob,
        prompts_file,
        storage=storage,
        teacher_callable=lambda _prompt, _cfg: {
            "response": "OK",
            "topk_logprobs": [
                {
                    "top_candidates": [
                        {"token": "A", "log_probability": -0.1},
                        {"token": "B", "log_probability": -1.2},
                        {"token": "C", "log_probability": -2.5},
                    ]
                }
            ],
        },
    )
    assert storage.exists(f"{ws.task_uri}/task_state.json")


def test_init_warns_and_exits_when_gcs_not_empty_unless_force_reset(
    tmp_path: Path,
) -> None:
    from click.testing import CliRunner

    from distillfw.cli import main
    from distillfw.state import TaskWorkspaceExistsError

    tasks_root = tmp_path / "tasks_root"

    cfg = DistillationConfig(
        task_id="existing-task-check",
        local=LocalConfig(storage_root=str(tasks_root)),
    )
    config_file = tmp_path / "config.yaml"
    config_file.write_text(cfg.to_yaml(), encoding="utf-8")

    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(json.dumps({"prompt": "Hello"}) + "\n", encoding="utf-8")

    runner = CliRunner()

    # First init succeeds on empty target path
    res1 = runner.invoke(main, ["init", "--config", str(config_file), "--prompts", str(prompts_file)])
    assert res1.exit_code == 0, res1.output
    assert "Initialized task workspace" in res1.output

    # Write a sentinel artifact into the workspace to ensure existing files are untouched on re-init
    storage = StorageBackend()
    sentinel_uri = f"{cfg.task_uri}/01_raw_dataset/sentinel.txt"
    storage.write_text(sentinel_uri, "do-not-delete")

    # Second init without --force-reset warns and exits without modifying anything
    res2 = runner.invoke(main, ["init", "--config", str(config_file), "--prompts", str(prompts_file)])
    assert res2.exit_code == 0, res2.output
    assert "Warning:" in res2.output
    assert "--force-reset" in res2.output
    assert storage.read_text(sentinel_uri) == "do-not-delete"

    # Programmatic API raises TaskWorkspaceExistsError when force_reset=False
    with pytest.raises(TaskWorkspaceExistsError):
        TaskWorkspace.initialize(cfg, prompts_file, storage=storage, force_reset=False)
    assert storage.read_text(sentinel_uri) == "do-not-delete"

    # Init WITH --force-reset when user answers 'n' (declines confirmation) -> cancels without erasing
    res_decline = runner.invoke(
        main,
        ["init", "--config", str(config_file), "--prompts", str(prompts_file), "--force-reset"],
        input="n\n",
    )
    assert res_decline.exit_code == 0, res_decline.output
    assert "COMPLETELY AND PERMANENTLY ERASE" in res_decline.output
    assert "Cancelled:" in res_decline.output
    assert storage.read_text(sentinel_uri) == "do-not-delete"

    # Init WITH --force-reset when user answers 'y' (confirms) -> completely erases and re-initializes
    res3 = runner.invoke(
        main,
        ["init", "--config", str(config_file), "--prompts", str(prompts_file), "--force-reset"],
        input="y\n",
    )
    assert res3.exit_code == 0, res3.output
    assert "COMPLETELY AND PERMANENTLY ERASE" in res3.output
    assert "Initialized task workspace" in res3.output
    assert not storage.exists(sentinel_uri)
    assert storage.exists(f"{cfg.task_uri}/task_state.json")


def test_activity_logging_local_timestamped_file_and_periodic_gcs_upload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import re
    import time

    from distillfw.logging_utils import TaskRunLogger, get_logger

    storage = StorageBackend(local_root=tmp_path / "gcs_root")
    local_log_dir = tmp_path / "local_logs"
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(json.dumps({"prompt": "Hello"}) + "\n", encoding="utf-8")

    cfg = DistillationConfig(
        task_id="log-sync-task",
        gcp=GCPConfig(project_id="p", bucket_name="b"),
    )
    ws = TaskWorkspace.initialize(cfg, prompts_file, storage=storage)

    with TaskRunLogger(
        command_name="resume",
        workspace=ws,
        log_dir=local_log_dir,
        sync_interval_seconds=0.05,
    ) as run_logger:
        assert re.match(r"^\d{8}_\d{6}_resume\.log$", run_logger.filename)
        assert run_logger.local_log_path.exists()
        get_logger("stages.generator").info("Processing shard 1/3...")
        time.sleep(0.15)
        # Background thread should have uploaded at least once while still running
        assert run_logger.upload_count >= 1
        assert storage.exists(run_logger.gcs_log_uri)
        mid_gcs_text = storage.read_text(run_logger.gcs_log_uri)
        assert "Processing shard 1/3..." in mid_gcs_text
        get_logger("stages.trainer").info("Training epoch 1 completed.")

    # After context exit, final log upload must include completion message and latest stage logs
    captured = capsys.readouterr()
    assert "Processing shard 1/3..." in captured.err
    assert "Finished command 'resume'" in captured.err

    final_gcs_text = storage.read_text(run_logger.gcs_log_uri)
    assert "Processing shard 1/3..." in final_gcs_text
    assert "Training epoch 1 completed." in final_gcs_text
    assert "Finished command 'resume'" in final_gcs_text


def test_gemini_request_retries_at_least_10_times_with_increasing_delays() -> None:
    from distillfw.gcp.retry import (
        call_with_exponential_backoff,
        compute_increasing_backoff_delays,
    )

    delays = compute_increasing_backoff_delays(max_retries=12)
    assert len(delays) >= 10
    assert all(delays[i] < delays[i + 1] for i in range(len(delays) - 1))

    recorded_sleeps: list[float] = []
    attempts = 0

    def flaky_gemini_call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts <= 10:
            raise RuntimeError(
                "429 RESOURCE_EXHAUSTED: extensible_stubs::OVERLOADED_TOO_MANY_RETRIES_PER_REQUEST"
            )
        return "Recovered on attempt 11"

    result = call_with_exponential_backoff(
        flaky_gemini_call,
        max_retries=12,
        sleep_fn=lambda s: recorded_sleeps.append(s),
    )
    assert result == "Recovered on attempt 11"
    assert attempts == 11
    assert len(recorded_sleeps) == 10
    # Verify every consecutive retry wait time is strictly increasing
    assert all(recorded_sleeps[i] < recorded_sleeps[i + 1] for i in range(len(recorded_sleeps) - 1))


def test_init_preflight_checks_gcp_bucket_region_match(tmp_path: Path) -> None:
    from distillfw.state import TaskInitializationError

    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(json.dumps({"prompt": "Hello"}) + "\n", encoding="utf-8")

    cfg = DistillationConfig(
        task_id="region-check-task",
        gcp=GCPConfig(
            project_id="distillfw",
            location="us-central1",
            bucket_name="distillfw-storage",
        ),
    )

    class MockRegionStorage(StorageBackend):
        def __init__(self, location_response: str | Exception) -> None:
            super().__init__(local_root=tmp_path)
            self._location_response = location_response

        def get_bucket_location(self, bucket_name: str) -> str | None:
            if isinstance(self._location_response, Exception):
                raise self._location_response
            return self._location_response

    # 1. Bucket in multi-region 'US' while gcp.location='us-central1' -> TaskInitializationError with gcloud instructions
    with pytest.raises(TaskInitializationError) as exc_mismatch:
        TaskWorkspace.initialize(cfg, prompts_file, storage=MockRegionStorage("US"))
    msg = str(exc_mismatch.value)
    assert "location 'us'" in msg
    assert "does not match gcp.location='us-central1'" in msg
    assert "gcloud storage buckets delete gs://distillfw-storage --project=distillfw" in msg
    assert "gcloud storage buckets create" in msg
    assert "--location=us-central1" in msg
    assert "--uniform-bucket-level-access" in msg

    # 2. Missing bucket -> TaskInitializationError with bucket creation command
    with pytest.raises(TaskInitializationError) as exc_missing:
        TaskWorkspace.initialize(
            cfg, prompts_file, storage=MockRegionStorage(FileNotFoundError("Not found"))
        )
    msg_missing = str(exc_missing.value)
    assert "does not exist" in msg_missing
    assert "gcloud storage buckets create gs://distillfw-storage" in msg_missing
    assert "--location=us-central1" in msg_missing

    # 3. Matching regional bucket ('US-CENTRAL1' vs 'us-central1') -> succeeds
    ws = TaskWorkspace.initialize(cfg, prompts_file, storage=MockRegionStorage("US-CENTRAL1"))
    assert ws.storage.exists(ws.state_uri)


def test_upstream_stage_prerequisite_and_vertex_job_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from distillfw.gcp.vertex import VertexJobManager
    from distillfw.pipeline import DistillationPipeline
    from distillfw.stages.trainer import ModelTrainer

    storage = StorageBackend(local_root=tmp_path)
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(json.dumps({"prompt": "Hello"}) + "\n", encoding="utf-8")

    cfg = DistillationConfig(
        task_id="prerequisite-test",
        gcp=GCPConfig(project_id="p", bucket_name="b"),
    )
    ws = TaskWorkspace.initialize(cfg, prompts_file, storage=storage)
    ws.mark_stage_completed(StageName.DATASET_GENERATOR)
    ws.mark_stage_completed(StageName.DATASET_FORMATTER)
    ws.mark_stage_running(
        StageName.MODEL_TRAINER,
        cursor_updates={"vertex_job_resource_name": "projects/p/locations/us-central1/customJobs/123"},
    )

    pipeline = DistillationPipeline(workspace=ws)

    # Attempting to run model_evaluator while model_trainer is still RUNNING must be blocked immediately
    with pytest.raises(
        RuntimeError,
        match="Cannot execute stage 'model_evaluator' because upstream prerequisite stage 'model_trainer' is currently RUNNING",
    ):
        pipeline.run_stage(StageName.MODEL_EVALUATOR)

    # Verify model_evaluator was NOT marked FAILED by the blocked attempt
    assert ws.load_state().stages[StageName.MODEL_EVALUATOR].status == StageStatus.PENDING

    # Verify ModelTrainer in vertex_custom_job mode polls until JOB_STATE_SUCCEEDED
    poll_states = [
        {"job_resource_name": "jobs/123", "display_name": "j", "state": "JOB_STATE_PENDING", "error": None},
        {"job_resource_name": "jobs/123", "display_name": "j", "state": "JOB_STATE_RUNNING", "error": None},
        {"job_resource_name": "jobs/123", "display_name": "j", "state": "JOB_STATE_SUCCEEDED", "error": None},
    ]
    poll_idx = 0

    def fake_get_job_status(self, job_resource_name: str):
        nonlocal poll_idx
        res = poll_states[min(poll_idx, len(poll_states) - 1)]
        poll_idx += 1
        return res

    monkeypatch.setattr(VertexJobManager, "get_job_status", fake_get_job_status)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    trainer = ModelTrainer(workspace=ws)
    res = trainer.run()
    assert res["normalized_state"] == "JOB_STATE_SUCCEEDED"
    assert poll_idx == 3
    assert ws.load_state().stages[StageName.MODEL_TRAINER].status == StageStatus.COMPLETED


def test_status_cmd_refreshes_vertex_training_job_and_persists_to_gcs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    from distillfw.cli import main
    from distillfw.gcp.vertex import VertexJobManager

    storage = StorageBackend(local_root=tmp_path)
    monkeypatch.setattr("distillfw.state.StorageBackend", lambda *a, **kw: storage)

    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(json.dumps({"prompt": "Hello"}) + "\n", encoding="utf-8")

    cfg = DistillationConfig(
        task_id="status-refresh-test",
        gcp=GCPConfig(project_id="p", bucket_name="b"),
    )
    ws = TaskWorkspace.initialize(cfg, prompts_file, storage=storage)
    ws.mark_stage_completed(StageName.DATASET_GENERATOR)
    ws.mark_stage_completed(StageName.DATASET_FORMATTER)
    ws.mark_stage_running(
        StageName.MODEL_TRAINER,
        cursor_updates={
            "vertex_job_resource_name": "projects/123/locations/us-central1/customJobs/999"
        },
    )

    runner = CliRunner()

    # Case 1: Vertex AI job is currently JOB_STATE_RUNNING -> updates cursor and persists to GCS
    monkeypatch.setattr(
        VertexJobManager,
        "get_job_status",
        lambda self, name: {
            "job_resource_name": name,
            "display_name": "train-job",
            "state": "JobState.JOB_STATE_RUNNING",
            "error": None,
        },
    )
    res_running = runner.invoke(main, ["status", cfg.task_uri])
    assert res_running.exit_code == 0, res_running.output
    assert "JOB_STATE_RUNNING" in res_running.output
    persisted_running = ws.load_state()
    assert persisted_running.stages[StageName.MODEL_TRAINER].status == StageStatus.RUNNING
    assert (
        persisted_running.stages[StageName.MODEL_TRAINER].progress_cursor["vertex_job_state"]
        == "JOB_STATE_RUNNING"
    )

    # Case 2: Vertex AI job finishes with JOB_STATE_SUCCEEDED -> transitions stage to COMPLETED & persists to GCS
    monkeypatch.setattr(
        VertexJobManager,
        "get_job_status",
        lambda self, name: {
            "job_resource_name": name,
            "display_name": "train-job",
            "state": "JobState.JOB_STATE_SUCCEEDED",
            "error": None,
        },
    )
    res_done = runner.invoke(main, ["status", cfg.task_uri])
    assert res_done.exit_code == 0, res_done.output
    assert "JOB_STATE_SUCCEEDED" in res_done.output
    persisted_done = ws.load_state()
    assert persisted_done.stages[StageName.MODEL_TRAINER].status == StageStatus.COMPLETED
    assert (
        persisted_done.stages[StageName.MODEL_TRAINER].progress_cursor["vertex_job_state"]
        == "JOB_STATE_SUCCEEDED"
    )

    # Case 3: Vertex AI job returns integer state 5 (JobState.JOB_STATE_FAILED) -> transitions stage to FAILED
    ws.mark_stage_running(
        StageName.MODEL_TRAINER,
        cursor_updates={
            "vertex_job_resource_name": "projects/123/locations/us-central1/customJobs/999"
        },
    )
    monkeypatch.setattr(
        VertexJobManager,
        "get_job_status",
        lambda self, name: {
            "job_resource_name": name,
            "display_name": "train-job",
            "state": "5",
            "normalized_state": "JOB_STATE_FAILED",
            "error": "StartError: exec: distillfw: executable file not found",
        },
    )
    res_failed = runner.invoke(main, ["status", cfg.task_uri])
    assert res_failed.exit_code == 0, res_failed.output
    assert "JOB_STATE_FAILED" in res_failed.output
    persisted_failed = ws.load_state()
    assert persisted_failed.stages[StageName.MODEL_TRAINER].status == StageStatus.FAILED
    assert (
        persisted_failed.stages[StageName.MODEL_TRAINER].progress_cursor["vertex_job_state"]
        == "JOB_STATE_FAILED"
    )


def test_normalize_vertex_job_state_handles_int_and_enum_strings() -> None:
    from distillfw.gcp.vertex import normalize_vertex_job_state

    assert normalize_vertex_job_state(3) == "JOB_STATE_RUNNING"
    assert normalize_vertex_job_state("3") == "JOB_STATE_RUNNING"
    assert normalize_vertex_job_state(4) == "JOB_STATE_SUCCEEDED"
    assert normalize_vertex_job_state("4") == "JOB_STATE_SUCCEEDED"
    assert normalize_vertex_job_state(5) == "JOB_STATE_FAILED"
    assert normalize_vertex_job_state("5") == "JOB_STATE_FAILED"
    assert normalize_vertex_job_state("JobState.JOB_STATE_FAILED") == "JOB_STATE_FAILED"
    assert normalize_vertex_job_state("JOB_STATE_CANCELLED") == "JOB_STATE_CANCELLED"


def test_trainer_resubmits_new_job_when_previous_vertex_job_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from distillfw.gcp.vertex import VertexJobManager
    from distillfw.stages.trainer import ModelTrainer

    storage = StorageBackend(local_root=tmp_path)
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(json.dumps({"prompt": "Hello"}) + "\n", encoding="utf-8")

    cfg = DistillationConfig(
        task_id="trainer-resubmit-test",
        gcp=GCPConfig(project_id="p", bucket_name="b"),
    )
    ws = TaskWorkspace.initialize(cfg, prompts_file, storage=storage)
    ws.mark_stage_completed(StageName.DATASET_GENERATOR)
    ws.mark_stage_completed(StageName.DATASET_FORMATTER)
    # Record a previously failed job ID in cursor
    ws.mark_stage_running(
        StageName.MODEL_TRAINER,
        cursor_updates={
            "vertex_job_resource_name": "projects/123/locations/us-central1/customJobs/old_failed_job",
            "vertex_job_state": "JOB_STATE_FAILED",
        },
    )

    submitted_jobs: list[str] = []

    def fake_get_job_status(self, job_resource_name: str) -> dict:
        if job_resource_name.endswith("old_failed_job"):
            return {
                "job_resource_name": job_resource_name,
                "display_name": "old-job",
                "state": "5",
                "normalized_state": "JOB_STATE_FAILED",
                "error": "StartError",
            }
        return {
            "job_resource_name": job_resource_name,
            "display_name": "new-job",
            "state": "4",
            "normalized_state": "JOB_STATE_SUCCEEDED",
            "error": None,
        }

    def fake_submit_training_job(self, display_name: str, task_uri: str, training_config, env_vars=None) -> dict:
        new_name = "projects/123/locations/us-central1/customJobs/new_succeeded_job"
        submitted_jobs.append(new_name)
        return {
            "job_resource_name": new_name,
            "display_name": display_name,
            "state": "3",
            "normalized_state": "JOB_STATE_RUNNING",
        }

    monkeypatch.setattr(VertexJobManager, "get_job_status", fake_get_job_status)
    monkeypatch.setattr(VertexJobManager, "submit_training_job", fake_submit_training_job)

    trainer = ModelTrainer(ws)
    result = trainer.run()
    assert len(submitted_jobs) == 1
    assert result["normalized_state"] == "JOB_STATE_SUCCEEDED"
    assert ws.load_state().stages[StageName.MODEL_TRAINER].status == StageStatus.COMPLETED


def test_hf_token_required_on_init_and_model_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    from distillfw.cli import main
    from distillfw.stages.trainer import ModelTrainer
    from distillfw.state import TaskInitializationError

    tasks_root = tmp_path / "tasks_root"
    cfg = DistillationConfig(
        task_id="hf-token-check",
        local=LocalConfig(storage_root=str(tasks_root)),
    )
    config_file = tmp_path / "config.yaml"
    config_file.write_text(cfg.to_yaml(), encoding="utf-8")

    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(json.dumps({"prompt": "Hello"}) + "\n", encoding="utf-8")

    runner = CliRunner()

    # 1. distillfw init fails with exit code 1 and clear instructions when HF_TOKEN is unset or empty
    for bad_val in (None, "", "   "):
        if bad_val is None:
            monkeypatch.delenv("HF_TOKEN", raising=False)
        else:
            monkeypatch.setenv("HF_TOKEN", bad_val)

        res_init = runner.invoke(
            main, ["init", "--config", str(config_file), "--prompts", str(prompts_file)]
        )
        assert res_init.exit_code == 1, res_init.output
        assert "HF_TOKEN" in res_init.output
        assert "export HF_TOKEN=" in res_init.output

    # 2. Initialize workspace with valid HF_TOKEN, then unset HF_TOKEN before starting model_trainer
    monkeypatch.setenv("HF_TOKEN", "hf_valid_token_abc")
    res_ok = runner.invoke(
        main, ["init", "--config", str(config_file), "--prompts", str(prompts_file)]
    )
    assert res_ok.exit_code == 0, res_ok.output

    ws = TaskWorkspace(task_uri=cfg.task_uri)
    ws.mark_stage_completed(StageName.DATASET_GENERATOR)
    ws.mark_stage_completed(StageName.DATASET_FORMATTER)

    monkeypatch.delenv("HF_TOKEN", raising=False)
    trainer = ModelTrainer(ws)
    with pytest.raises(TaskInitializationError, match="HF_TOKEN"):
        trainer.run()

    res_stage = runner.invoke(main, ["run-stage", cfg.task_uri, "--stage", "model_trainer"])
    assert res_stage.exit_code == 1, res_stage.output
    assert "HF_TOKEN" in res_stage.output
    assert "export HF_TOKEN=" in res_stage.output


def test_reset_training_and_reset_eval_cli_commands(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from distillfw.cli import main

    tasks_root = tmp_path / "tasks_root"
    cfg = DistillationConfig(
        task_id="reset-stages-test",
        local=LocalConfig(storage_root=str(tasks_root)),
    )
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(json.dumps({"prompt": "Hello"}) + "\n", encoding="utf-8")

    ws = TaskWorkspace.initialize(cfg, prompts_file)
    ws.mark_stage_completed(StageName.DATASET_GENERATOR)
    ws.mark_stage_completed(StageName.DATASET_FORMATTER)
    ws.mark_stage_failed(StageName.MODEL_TRAINER, "OOM failure")
    ws.mark_stage_failed(StageName.MODEL_EVALUATOR, "Tokenizer failure")

    runner = CliRunner()

    # 1. Reset training stage via CLI
    res_train = runner.invoke(main, ["reset-training", cfg.task_uri])
    assert res_train.exit_code == 0, res_train.output
    assert "model_trainer" in res_train.output
    state_after_train_reset = ws.load_state()
    assert state_after_train_reset.stages[StageName.MODEL_TRAINER].status == StageStatus.PENDING
    assert state_after_train_reset.stages[StageName.MODEL_TRAINER].attempt_count == 0
    assert state_after_train_reset.stages[StageName.MODEL_TRAINER].progress_cursor == {}
    assert state_after_train_reset.stages[StageName.MODEL_TRAINER].error_message is None

    # 2. Reset eval stage via CLI
    res_eval = runner.invoke(main, ["reset-eval", cfg.task_uri])
    assert res_eval.exit_code == 0, res_eval.output
    assert "model_evaluator" in res_eval.output
    state_after_eval_reset = ws.load_state()
    assert state_after_eval_reset.stages[StageName.MODEL_EVALUATOR].status == StageStatus.PENDING
    assert state_after_eval_reset.stages[StageName.MODEL_EVALUATOR].attempt_count == 0
    assert state_after_eval_reset.stages[StageName.MODEL_EVALUATOR].progress_cursor == {}
    assert state_after_eval_reset.stages[StageName.MODEL_EVALUATOR].error_message is None
    assert state_after_eval_reset.status == StageStatus.PENDING


def test_judge_model_access_preflight_in_init_and_evaluator_and_location_config(
    tmp_path: Path,
) -> None:
    from click.testing import CliRunner

    from distillfw.cli import main
    from distillfw.config import EvaluationConfig
    from distillfw.stages.evaluator import ModelEvaluator
    from distillfw.state import TaskInitializationError

    tasks_root = tmp_path / "tasks_root"
    cfg = DistillationConfig(
        task_id="judge-access-check",
        local=LocalConfig(storage_root=str(tasks_root)),
        evaluation=EvaluationConfig(
            judge_model_id="gemini-3.5-pro",
            judge_model_location="global",
        ),
    )
    assert cfg.evaluation.resolved_judge_location == "global"

    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(json.dumps({"prompt": "Hello"}) + "\n", encoding="utf-8")

    def failing_judge(_p: str, _r: str, _s: str):
        raise RuntimeError(
            "404 NOT_FOUND: Publisher model `projects/distillfw/locations/global/publishers/google/models/gemini-3.5-pro` was not found"
        )

    # 1. TaskWorkspace.initialize fails when judge_callable raises 404 NOT_FOUND
    with pytest.raises(TaskInitializationError, match="Judge model access error"):
        TaskWorkspace.initialize(cfg, prompts_file, judge_callable=failing_judge)

    # 2. Initialize workspace with accessible judge, then verify ModelEvaluator.run checks judge access before running
    ws = TaskWorkspace.initialize(
        cfg,
        prompts_file,
        judge_callable=lambda _p, _r, _s: {"score": 5, "reason": "ok"},
    )
    ws.mark_stage_completed(StageName.DATASET_GENERATOR)
    ws.mark_stage_completed(StageName.DATASET_FORMATTER)
    ws.mark_stage_completed(StageName.MODEL_TRAINER)

    evaluator = ModelEvaluator(ws, judge_fn=failing_judge)
    with pytest.raises(TaskInitializationError, match="judge_model_location"):
        evaluator.run()
    failed_state = ws.load_state()
    assert failed_state.stages[StageName.MODEL_EVALUATOR].status == StageStatus.FAILED
    assert failed_state.status == StageStatus.FAILED

    # 3. Verify distillfw reset-eval --config updates the frozen config on GCS and resets model_evaluator
    updated_cfg = DistillationConfig(
        task_id="judge-access-check",
        local=LocalConfig(storage_root=str(tasks_root)),
        evaluation=EvaluationConfig(
            judge_model_id="gemini-3.5-flash",
            judge_model_location="us-central1",
        ),
    )
    updated_config_file = tmp_path / "updated_config.yaml"
    updated_config_file.write_text(updated_cfg.to_yaml(), encoding="utf-8")

    runner = CliRunner()
    res_reset = runner.invoke(
        main,
        ["reset-eval", cfg.task_uri, "--config", str(updated_config_file)],
    )
    assert res_reset.exit_code == 0, res_reset.output
    reloaded_cfg = ws.load_config()
    assert reloaded_cfg.evaluation.judge_model_id == "gemini-3.5-flash"
    assert reloaded_cfg.evaluation.judge_model_location == "us-central1"
    assert ws.load_state().stages[StageName.MODEL_EVALUATOR].status == StageStatus.PENDING

    # 4. Verify verify_judge_model_access live genai.Client path (judge_callable=None) uses valid TeacherConfig attributes
    import google.genai as genai

    from distillfw.state import verify_judge_model_access

    class _FakeResponse:
        text = "OK"

    class _FakeChat:
        def send_message(self, message: str):
            return _FakeResponse()

    class _FakeChats:
        def create(self, model: str, config=None):
            return _FakeChat()

    class _FakeGenAIClient:
        def __init__(self, **kwargs):
            self.chats = _FakeChats()

    gcp_cfg = DistillationConfig(
        task_id="judge-live-client-test",
        gcp=GCPConfig(project_id="distillfw", bucket_name="b", location="us-central1"),
        evaluation=EvaluationConfig(
            judge_model_id="gemini-3.5-flash",
            judge_model_location="global",
        ),
    )
    orig_client = genai.Client
    genai.Client = _FakeGenAIClient  # type: ignore[assignment]
    try:
        verify_judge_model_access(gcp_cfg, judge_callable=None)
    finally:
        genai.Client = orig_client  # type: ignore[assignment]









