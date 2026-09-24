"""Tests for the distillfw Web UI backend controller and HTTP endpoints."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from urllib.request import urlopen

import pandas as pd

from distillfw.gcp.storage import StorageBackend
from distillfw.state import StageName, StageStatus, TaskState, TaskWorkspace
from distillfw.ui.server import TaskUIController, create_ui_server


def _seed_sample_task_on_storage(storage: StorageBackend, root_uri: str, task_id: str) -> str:
    task_uri = f"{root_uri.rstrip('/')}/{task_id}"
    ws = TaskWorkspace(task_uri=task_uri, storage=storage)

    sample_config_yaml = f"""task_id: {task_id}
description: Sample distillation task for UI testing
gcp:
  project_id: demo-gcp-project
  location: us-central1
  bucket_name: demo-bucket
  tasks_prefix: tasks
teacher:
  model_id: gemini-3.5-flash
  location: global
  response_logprobs: false
student:
  model_id: google/gemma-3-4b-it
  peft_method: lora
formatting:
  prompt_format: chat
  dataset_representation: text
  train_split_ratio: 0.8
  val_split_ratio: 0.1
  test_split_ratio: 0.1
training:
  paradigm: off_policy
  algorithm: sft_seqkd
  execution_mode: vertex_custom_job
  vertex_machine_type: g2-standard-48
  vertex_accelerator_type: NVIDIA_L4
  vertex_accelerator_count: 4
evaluation:
  metrics: [rouge, bleu, exact_match, llm_judge]
  judge_model_id: gemini-3.5-flash
deployment:
  machine_type: g2-standard-12
  accelerator_type: NVIDIA_L4
  accelerator_count: 1
  min_replica_count: 1
  max_replica_count: 2
"""
    storage.write_text(ws.config_uri, sample_config_yaml)

    # Create task_state.json with realistic GCP resource IDs
    state = TaskState(
        task_id=task_id,
        task_uri=task_uri,
        config_sha256="abc12345",
        status=StageStatus.COMPLETED,
        current_stage=StageName.MODEL_DEPLOYER,
    )
    state.stages[StageName.DATASET_GENERATOR].status = StageStatus.COMPLETED
    state.stages[StageName.DATASET_FORMATTER].status = StageStatus.COMPLETED
    state.stages[StageName.MODEL_TRAINER].status = StageStatus.COMPLETED
    state.stages[StageName.MODEL_TRAINER].progress_cursor = {
        "vertex_job_resource_name": "projects/demo-gcp-project/locations/us-central1/customJobs/987654321",
        "vertex_job_state": "JOB_STATE_SUCCEEDED",
    }
    state.stages[StageName.MODEL_EVALUATOR].status = StageStatus.COMPLETED
    state.stages[StageName.MODEL_DEPLOYER].status = StageStatus.COMPLETED
    ws.save_state(state)

    # Seed input prompts (.jsonl)
    storage.write_text(
        f"{ws.inputs_dir_uri}/prompts.jsonl",
        "\n".join(
            [
                json.dumps({"data": {"prompt": "Classify ticket: VPN login error"}, "metadata": {"priority": "high"}}),
                json.dumps({"data": {"prompt": "Classify ticket: Billing invoice question"}, "metadata": {"priority": "low"}}),
            ]
        )
        + "\n",
    )

    # Seed formatted dataset (.parquet)
    df_train = pd.DataFrame(
        [
            {"prompt": "Classify ticket: VPN login error", "completion": "{\"category\": \"network\"}", "split": "train"},
            {"prompt": "Classify ticket: Billing invoice question", "completion": "{\"category\": \"billing\"}", "split": "train"},
        ]
    )
    parquet_path = storage._resolve_local_path(f"{ws.formatted_dataset_dir_uri}/train.parquet")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(parquet_path, index=False)

    # Seed log file
    storage.write_text(
        f"{ws.logs_dir_uri}/20260923_060000_resume.log",
        "2026-09-23 06:00:00 | INFO | distillfw | Starting pipeline resume\n"
        "2026-09-23 06:05:00 | INFO | distillfw | Completed evaluation stage\n",
    )

    # Seed evaluation scorecard & side-by-side predictions
    scorecard = {
        "task_id": task_id,
        "teacher_model": "gemini-3.5-flash",
        "student_model": "google/gemma-3-4b-it",
        "algorithm": "sft_seqkd",
        "splits": {
            "test": {
                "num_samples": 2,
                "before_training": {
                    "lexical_metrics": {"exact_match": 0.0, "rouge1": 0.42, "rouge2": 0.20, "bleu": 0.15},
                    "llm_judge": {"mean_rubric_score": 2.5, "win_or_tie_rate_vs_teacher": 0.25},
                },
                "after_training": {
                    "lexical_metrics": {"exact_match": 0.5, "rouge1": 0.85, "rouge2": 0.72, "bleu": 0.68},
                    "llm_judge": {"mean_rubric_score": 4.6, "win_or_tie_rate_vs_teacher": 0.90},
                },
                "improvement": {
                    "lexical_metrics": {"exact_match": 0.5, "rouge1": 0.43, "rouge2": 0.52, "bleu": 0.53},
                    "llm_judge": {"mean_rubric_score": 2.1, "win_or_tie_rate_vs_teacher": 0.65},
                },
            }
        },
    }
    storage.write_json(f"{ws.evaluation_dir_uri}/scorecard.json", scorecard)

    predictions = [
        {
            "split": "test",
            "prompt": "Classify ticket: VPN login error",
            "base_student_prediction": "Unstructured response: maybe network issue.",
            "distilled_student_prediction": "{\"category\": \"network\", \"severity\": \"P1\"}",
            "teacher_reference": "{\"category\": \"network\", \"severity\": \"P1\"}",
            "base_latency_ms": 42.3,
            "distilled_latency_ms": 38.1,
        },
        {
            "split": "train",
            "prompt": "Classify ticket: Billing invoice question",
            "base_student_prediction": "Ask accounting department.",
            "distilled_student_prediction": "{\"category\": \"billing\", \"severity\": \"P3\"}",
            "teacher_reference": "{\"category\": \"billing\", \"severity\": \"P3\"}",
            "base_latency_ms": 39.0,
            "distilled_latency_ms": 36.4,
        },
    ]
    storage.write_text(
        f"{ws.evaluation_dir_uri}/predictions.jsonl",
        "\n".join(json.dumps(p) for p in predictions) + "\n",
    )

    # Seed deployment endpoint_info.json
    storage.write_json(
        f"{ws.deployment_dir_uri}/endpoint_info.json",
        {
            "model_resource_name": "projects/demo-gcp-project/locations/us-central1/models/11223344",
            "endpoint_resource_name": "projects/demo-gcp-project/locations/us-central1/endpoints/55667788",
            "endpoint_display_name": f"distillfw-{task_id}",
        },
    )
    return task_uri


def test_ui_controller_end_to_end(tmp_path: Path) -> None:
    storage = StorageBackend(local_root=tmp_path)
    root_uri = "gs://demo-bucket/tasks"
    task_uri = _seed_sample_task_on_storage(storage, root_uri, "ticket-classifier-v1")

    controller = TaskUIController(storage=storage, default_root_uri=root_uri)

    # 1. List tasks
    tasks_resp = controller.list_tasks(root_uri)
    assert tasks_resp["count"] == 1
    assert tasks_resp["tasks"][0]["task_id"] == "ticket-classifier-v1"
    assert tasks_resp["tasks"][0]["status"] == "COMPLETED"

    # 2. Task details + GCP resources
    details = controller.get_task_details(task_uri, refresh_vertex=False)
    assert details["task_id"] == "ticket-classifier-v1"
    assert details["status"] == "COMPLETED"
    assert len(details["logs"]) == 1
    assert len(details["datasets"]) >= 3
    assert details["has_predictions"] is True
    assert details["scorecard"] is not None

    resources_by_cat = {r["category"]: r for r in details["gcp_resources"]}
    assert "GCS Workspace" in resources_by_cat
    assert "console.cloud.google.com/storage/browser/demo-bucket/tasks/ticket-classifier-v1" in (
        resources_by_cat["GCS Workspace"]["console_url"] or ""
    )
    assert "Vertex AI Training" in resources_by_cat
    assert "987654321" in resources_by_cat["Vertex AI Training"]["resource_id"]
    assert "console.cloud.google.com/vertex-ai/locations/us-central1/training/987654321" in (
        resources_by_cat["Vertex AI Training"]["console_url"] or ""
    )
    assert "Vertex AI Model Registry" in resources_by_cat
    assert "11223344" in resources_by_cat["Vertex AI Model Registry"]["resource_id"]
    assert "Vertex AI Endpoint" in resources_by_cat
    assert "55667788" in resources_by_cat["Vertex AI Endpoint"]["resource_id"]

    # 3. Config inspection
    cfg_resp = controller.get_task_config(task_uri)
    assert "gemini-3.5-flash" in cfg_resp["yaml_text"]
    assert cfg_resp["parsed"]["task_id"] == "ticket-classifier-v1"

    # 4. Log inspection
    log_uri = details["logs"][0]["uri"]
    log_resp = controller.get_log_content(log_uri)
    assert log_resp["line_count"] == 2
    assert "Completed evaluation stage" in log_resp["content"]

    # 5. Structured dataset table inspection (.parquet & .jsonl)
    parquet_uri = f"{task_uri}/02_formatted_dataset/train.parquet"
    ds_resp = controller.get_dataset_table(parquet_uri, offset=0, limit=10, search_query="VPN")
    assert ds_resp["total_rows"] == 1
    assert "prompt" in ds_resp["columns"]
    assert "VPN login error" in ds_resp["rows"][0]["prompt"]

    # 6. Side-by-side evaluation inferences (including per-case metrics)
    inf_resp = controller.get_evaluation_inferences(task_uri, split_filter="test")
    assert inf_resp["total_count"] == 2
    assert inf_resp["filtered_count"] == 1
    rec = inf_resp["records"][0]
    assert rec["student_before_training"].startswith("Unstructured response")
    assert "\"category\": \"network\"" in rec["student_after_training"]
    assert "\"category\": \"network\"" in rec["teacher_model"]
    assert "base_metrics" in rec
    assert "distilled_metrics" in rec
    assert "metrics_delta" in rec
    assert rec["base_metrics"]["exact_match"] == 0.0
    assert rec["distilled_metrics"]["exact_match"] == 1.0
    assert rec["metrics_delta"]["exact_match"] == 1.0
    assert rec["base_metrics"]["latency_ms"] == 42.3
    assert rec["distilled_metrics"]["latency_ms"] == 38.1


def test_ui_http_server_endpoints(tmp_path: Path) -> None:
    storage = StorageBackend(local_root=tmp_path)
    root_uri = "gs://demo-bucket/tasks"
    task_uri = _seed_sample_task_on_storage(storage, root_uri, "http-ui-test-task")
    controller = TaskUIController(storage=storage, default_root_uri=root_uri)

    server = create_ui_server(host="127.0.0.1", port=0, controller=controller)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base_url = f"http://127.0.0.1:{port}"
        with urlopen(f"{base_url}/") as resp:
            html = resp.read().decode("utf-8")
            assert "distillfw" in html
            assert "gcs-root-input" in html
            assert 'id="about-distillation-btn"' in html
            assert "About Distillation" in html

        with urlopen(f"{base_url}/app.js") as resp:
            app_js = resp.read().decode("utf-8")
            assert "openAboutDistillationPanel" in app_js
            assert "Off-Policy Distillation" in app_js
            assert "On-Policy &amp; Hybrid Distillation" in app_js
            assert "Vocabulary Alignment" in app_js
            assert "response_logprobs" in app_js

        with urlopen(f"{base_url}/api/tasks?root_uri={root_uri}") as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            assert payload["count"] == 1
            assert payload["tasks"][0]["task_id"] == "http-ui-test-task"

        with urlopen(f"{base_url}/api/task/inferences?task_uri={task_uri}&split=all") as resp:
            inf_payload = json.loads(resp.read().decode("utf-8"))
            assert inf_payload["total_count"] == 2
    finally:
        server.shutdown()
        server.server_close()
