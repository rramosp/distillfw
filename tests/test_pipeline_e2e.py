"""End-to-end tests for the 5-stage distillation pipeline and stateless GCS resumption."""

from __future__ import annotations

import json
from pathlib import Path

from distillfw.config import DistillationConfig, GCPConfig, TeacherConfig
from distillfw.gcp.storage import StorageBackend
from distillfw.pipeline import DistillationPipeline
from distillfw.state import StageName, StageStatus


def test_full_pipeline_and_midway_stateless_resumption(tmp_path: Path) -> None:
    storage = StorageBackend(local_root=tmp_path)
    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(
        "\n".join(json.dumps({"prompt": f"Summarize doc #{i}"}) for i in range(20)) + "\n",
        encoding="utf-8",
    )

    cfg = DistillationConfig(
        task_id="e2e-distill-task",
        gcp=GCPConfig(project_id="test-proj", bucket_name="test-bucket"),
        teacher=TeacherConfig(shard_size=8),
    )

    # Phase 1: Run Stages 1 and 2 on Worker 1, then stop
    pipeline_worker_1 = DistillationPipeline.init_task(
        config=cfg,
        prompts_path=prompts_file,
        storage=storage,
        teacher_callable=lambda prompt, _: {
            "completion": f"Summary of {prompt}",
            "thought": f"Analyzing {prompt}",
        },
    )
    state_after_phase_1 = pipeline_worker_1.resume(stop_after=StageName.DATASET_FORMATTER)
    assert state_after_phase_1.stages[StageName.DATASET_GENERATOR].status == StageStatus.COMPLETED
    assert state_after_phase_1.stages[StageName.DATASET_FORMATTER].status == StageStatus.COMPLETED
    assert state_after_phase_1.stages[StageName.MODEL_TRAINER].status == StageStatus.PENDING

    # Verify 3 shards were created (20 prompts / shard_size=8 => 3 shards)
    gen_cursor = state_after_phase_1.stages[StageName.DATASET_GENERATOR].progress_cursor
    assert gen_cursor["completed_shards"] == [0, 1, 2]

    # Phase 2: Resume remaining stages (3, 4, 5) on Worker 2 using ONLY task_uri
    pipeline_worker_2 = DistillationPipeline.from_task_uri(
        task_uri="gs://test-bucket/tasks/e2e-distill-task",
        storage=storage,
        custom_train_fn=lambda train_p, ckpt_d, export_d: {
            "train_loss": 0.19,
            "global_step": 25,
        },
        student_predict_fn=lambda prompts: (
            [f"Summary of {p}" for p in prompts],
            [14.2 for _ in prompts],
        ),
        judge_fn=lambda p, ref, pred: {"score": 5, "reason": "Matches teacher"},
        deploy_fn=lambda task_id, model_uri, deploy_cfg: {
            "endpoint_resource_name": f"projects/test-proj/locations/us-central1/endpoints/{task_id}",
            "model_artifact_uri": model_uri,
        },
    )
    final_state = pipeline_worker_2.resume()

    assert final_state.status == StageStatus.COMPLETED
    for stage in StageName:
        assert final_state.stages[stage].status == StageStatus.COMPLETED

    # Verify evaluation scorecard and deployment manifest exist on GCS
    assert storage.exists("gs://test-bucket/tasks/e2e-distill-task/04_evaluation/scorecard.json")
    assert storage.exists("gs://test-bucket/tasks/e2e-distill-task/06_deployment/endpoint_info.json")
