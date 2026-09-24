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
        "\n".join(json.dumps({"data": {"prompt": f"Summarize doc #{i}"}}) for i in range(20)) + "\n",
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

    # Verify both train.parquet and test.parquet exist and are disjoint
    assert storage.exists("gs://test-bucket/tasks/e2e-distill-task/02_formatted_dataset/train.parquet")
    assert storage.exists("gs://test-bucket/tasks/e2e-distill-task/02_formatted_dataset/test.parquet")

    import pandas as pd

    train_df = pd.read_parquet(
        storage.download_file(
            "gs://test-bucket/tasks/e2e-distill-task/02_formatted_dataset/train.parquet",
            tmp_path / "dl_train.parquet",
        )
    )
    test_df = pd.read_parquet(
        storage.download_file(
            "gs://test-bucket/tasks/e2e-distill-task/02_formatted_dataset/test.parquet",
            tmp_path / "dl_test.parquet",
        )
    )
    assert len(train_df) >= 1
    assert len(test_df) >= 1
    assert set(train_df["prompt"]).isdisjoint(set(test_df["prompt"]))

    def _extract_raw_prompt(formatted_prompt: str) -> str:
        return (
            formatted_prompt.replace("<start_of_turn>user\n", "")
            .replace("<end_of_turn>\n<start_of_turn>model\n", "")
            .strip()
        )

    # Phase 2: Resume remaining stages (3, 4, 5) on Worker 2 using ONLY task_uri
    pipeline_worker_2 = DistillationPipeline.from_task_uri(
        task_uri="gs://test-bucket/tasks/e2e-distill-task",
        storage=storage,
        custom_train_fn=lambda train_p, ckpt_d, export_d: {
            "train_loss": 0.19,
            "global_step": 25,
        },
        base_student_predict_fn=lambda prompts: (
            ["Unrelated baseline answer" for _ in prompts],
            [15.0 for _ in prompts],
        ),
        student_predict_fn=lambda prompts: (
            [f"Summary of {_extract_raw_prompt(p)}" for p in prompts],
            [14.2 for _ in prompts],
        ),
        judge_fn=lambda p, ref, pred: (
            {"score": 5, "reason": "Matches teacher"}
            if pred.strip() == ref.strip()
            else {"score": 2, "reason": "Baseline miss"}
        ),
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

    scorecard = storage.read_json(
        "gs://test-bucket/tasks/e2e-distill-task/04_evaluation/scorecard.json"
    )
    for split_name in ("train", "test"):
        assert split_name in scorecard["before_training"]
        assert split_name in scorecard["after_training"]
        assert split_name in scorecard["improvement"]
        assert split_name in scorecard["splits"]

        before_em = scorecard["before_training"][split_name]["lexical_metrics"]["exact_match"]
        after_em = scorecard["after_training"][split_name]["lexical_metrics"]["exact_match"]
        delta_em = scorecard["improvement"][split_name]["lexical_metrics"]["exact_match"]
        assert before_em == 0.0
        assert after_em == 1.0
        assert delta_em == 1.0

        before_judge = scorecard["before_training"][split_name]["llm_judge"]["mean_rubric_score"]
        after_judge = scorecard["after_training"][split_name]["llm_judge"]["mean_rubric_score"]
        delta_judge = scorecard["improvement"][split_name]["llm_judge"]["mean_rubric_score"]
        assert before_judge == 2.0
        assert after_judge == 5.0
        assert delta_judge == 3.0

