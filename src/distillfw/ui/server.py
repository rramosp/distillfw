"""HTTP backend and data controller for the distillfw Web UI."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

import pandas as pd
import yaml

from distillfw.gcp.storage import StorageBackend, parse_gcs_uri
from distillfw.state import ORDERED_STAGES, StageName, TaskWorkspace

STATIC_DIR = Path(__file__).parent / "static"


def _parse_vertex_resource_name(resource_name: str | None) -> dict[str, str]:
    """Parse `projects/<proj>/locations/<loc>/<type>/<id>` into components."""
    if not resource_name or not isinstance(resource_name, str):
        return {}
    pattern = re.compile(
        r"^projects/(?P<project>[^/]+)/locations/(?P<location>[^/]+)/(?P<collection>[^/]+)/(?P<resource_id>[^/]+)$"
    )
    match = pattern.match(resource_name.strip())
    if not match:
        return {}
    return match.groupdict()


def _gcs_console_url(uri: str, project_id: str | None = None) -> str | None:
    """Build a Google Cloud Console Storage Browser link for a `gs://` URI."""
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        return None
    try:
        bucket, object_path = parse_gcs_uri(uri)
    except ValueError:
        return None
    clean_path = object_path.strip("/")
    base = f"https://console.cloud.google.com/storage/browser/{bucket}"
    if clean_path:
        base = f"{base}/{clean_path}"
    if project_id:
        return f"{base}?project={quote(project_id)}"
    return base


def _gcs_object_console_url(uri: str, project_id: str | None = None) -> str | None:
    """Build a Google Cloud Console Storage Object details link for a `gs://` file URI."""
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        return None
    try:
        bucket, object_path = parse_gcs_uri(uri)
    except ValueError:
        return None
    clean_path = object_path.lstrip("/")
    if not clean_path:
        return _gcs_console_url(uri, project_id=project_id)
    base = f"https://console.cloud.google.com/storage/browser/_details/{bucket}/{clean_path}"
    if project_id:
        return f"{base}?project={quote(project_id)}"
    return base


def _build_gcp_resources(
    ws: TaskWorkspace,
    state_dict: dict[str, Any],
    config_dict: dict[str, Any],
    endpoint_info: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build structured GCP resource references and console URLs for the task workspace view."""
    gcp_cfg = config_dict.get("gcp") or {}
    project_id = gcp_cfg.get("project_id")
    default_location = gcp_cfg.get("location") or "us-central1"

    stages = state_dict.get("stages") or {}
    trainer_stage = stages.get(StageName.MODEL_TRAINER.value) or {}
    trainer_cursor = trainer_stage.get("progress_cursor") or {}
    trainer_artifacts = trainer_stage.get("artifacts") or {}

    deployer_stage = stages.get(StageName.MODEL_DEPLOYER.value) or {}
    deployer_artifacts = deployer_stage.get("artifacts") or {}
    ep_info = endpoint_info or deployer_artifacts.get("endpoint_info") or {}

    resources: list[dict[str, Any]] = []

    # 1. GCS Task Root Workspace
    resources.append(
        {
            "category": "GCS Workspace",
            "label": "Task Workspace Root",
            "resource_id": ws.task_uri,
            "status": state_dict.get("status", "PENDING"),
            "console_url": _gcs_console_url(ws.task_uri, project_id=project_id),
            "description": "Single-source-of-truth GCS directory containing config, state manifest, datasets, checkpoints, and logs.",
        }
    )

    # 2. GCS Checkpoints Directory
    resources.append(
        {
            "category": "GCS Checkpoints",
            "label": "Trainer Checkpoints (03_checkpoints)",
            "resource_id": ws.checkpoints_dir_uri,
            "status": trainer_stage.get("status", "PENDING"),
            "console_url": _gcs_console_url(ws.checkpoints_dir_uri, project_id=project_id),
            "description": "Resumable single-node training checkpoints and trainer state synced during Stage 3.",
        }
    )

    # 3. GCS Exported Model Directory
    exported_uri = trainer_artifacts.get("exported_model_uri") or ws.exported_model_dir_uri
    resources.append(
        {
            "category": "GCS Model Artifacts",
            "label": "Exported Student Model (05_exported_model)",
            "resource_id": exported_uri,
            "status": trainer_stage.get("status", "PENDING"),
            "console_url": _gcs_console_url(exported_uri, project_id=project_id),
            "description": "Merged Hugging Face / vLLM student model weights ready for evaluation and serving.",
        }
    )

    # 4. Vertex AI Custom Training Job
    vertex_job_name = (
        trainer_cursor.get("vertex_job_resource_name")
        or trainer_artifacts.get("vertex_job_resource_name")
    )
    parsed_job = _parse_vertex_resource_name(vertex_job_name)
    job_proj = parsed_job.get("project") or project_id
    job_loc = parsed_job.get("location") or default_location
    job_id = parsed_job.get("resource_id")
    job_console_url = None
    job_logs_url = None
    if job_id and job_proj and job_loc:
        job_console_url = (
            f"https://console.cloud.google.com/vertex-ai/locations/{quote(job_loc)}"
            f"/training/{quote(job_id)}/cpu?project={quote(job_proj)}"
        )
        log_query = f'resource.type="ml_job"\nresource.labels.job_id="{job_id}"'
        job_logs_url = (
            f"https://console.cloud.google.com/logs/query;query={quote(log_query)}"
            f"?project={quote(job_proj)}"
        )

    resources.append(
        {
            "category": "Vertex AI Training",
            "label": "Vertex AI Custom Training Job",
            "resource_id": vertex_job_name or "Not submitted yet",
            "status": trainer_cursor.get("vertex_job_state") or trainer_stage.get("status", "PENDING"),
            "console_url": job_console_url,
            "logs_url": job_logs_url,
            "description": (
                f"Machine: {config_dict.get('training', {}).get('vertex_machine_type', 'N/A')} | "
                f"Accelerator: {config_dict.get('training', {}).get('vertex_accelerator_type', 'N/A')} "
                f"x{config_dict.get('training', {}).get('vertex_accelerator_count', 1)}"
            ),
        }
    )

    # 4b. Vertex AI Custom Evaluation Job (when configured or submitted)
    eval_stage = stages.get("model_evaluator", {})
    eval_cursor = eval_stage.get("progress_cursor", {}) or {}
    eval_artifacts = eval_stage.get("artifacts", {}) or {}
    eval_cfg_dict = config_dict.get("evaluation", {}) or {}
    eval_vertex_job_name = (
        eval_cursor.get("vertex_job_resource_name")
        or eval_artifacts.get("vertex_job_resource_name")
    )
    eval_has_vertex_cfg = bool(eval_cfg_dict.get("vertex_machine_type"))
    if eval_vertex_job_name or eval_has_vertex_cfg:
        parsed_eval_job = _parse_vertex_resource_name(eval_vertex_job_name)
        eval_job_proj = parsed_eval_job.get("project") or project_id
        eval_job_loc = parsed_eval_job.get("location") or default_location
        eval_job_id = parsed_eval_job.get("resource_id")
        eval_job_console_url = None
        eval_job_logs_url = None
        if eval_job_id and eval_job_proj and eval_job_loc:
            eval_job_console_url = (
                f"https://console.cloud.google.com/vertex-ai/locations/{quote(eval_job_loc)}"
                f"/training/{quote(eval_job_id)}/cpu?project={quote(eval_job_proj)}"
            )
            eval_log_query = f'resource.type="ml_job"\nresource.labels.job_id="{eval_job_id}"'
            eval_job_logs_url = (
                f"https://console.cloud.google.com/logs/query;query={quote(eval_log_query)}"
                f"?project={quote(eval_job_proj)}"
            )
        resources.append(
            {
                "category": "Vertex AI Evaluation",
                "label": "Vertex AI Custom Evaluation Job",
                "resource_id": eval_vertex_job_name or "Not submitted yet",
                "status": eval_cursor.get("vertex_job_state") or eval_stage.get("status", "PENDING"),
                "console_url": eval_job_console_url,
                "logs_url": eval_job_logs_url,
                "description": (
                    f"Machine: {eval_cfg_dict.get('vertex_machine_type', 'Local')} | "
                    f"Accelerator: {eval_cfg_dict.get('vertex_accelerator_type', 'None')} "
                    f"x{eval_cfg_dict.get('vertex_accelerator_count', 0)}"
                ),
            }
        )


    # 5. Deployed Model in Vertex AI Model Registry
    model_res_name = ep_info.get("model_resource_name")
    parsed_model = _parse_vertex_resource_name(model_res_name)
    model_proj = parsed_model.get("project") or project_id
    model_loc = parsed_model.get("location") or default_location
    model_id = parsed_model.get("resource_id")
    model_console_url = None
    if model_id and model_proj and model_loc:
        model_console_url = (
            f"https://console.cloud.google.com/vertex-ai/locations/{quote(model_loc)}"
            f"/models/{quote(model_id)}?project={quote(model_proj)}"
        )

    resources.append(
        {
            "category": "Vertex AI Model Registry",
            "label": "Deployed Model Resource",
            "resource_id": model_res_name or "Not registered yet",
            "status": deployer_stage.get("status", "PENDING"),
            "console_url": model_console_url,
            "description": f"Base student: {config_dict.get('student', {}).get('model_id', 'N/A')}",
        }
    )

    # 6. Vertex AI Online Prediction Endpoint
    endpoint_res_name = ep_info.get("endpoint_resource_name") or ep_info.get("endpoint_name")
    parsed_ep = _parse_vertex_resource_name(endpoint_res_name)
    ep_proj = parsed_ep.get("project") or project_id
    ep_loc = parsed_ep.get("location") or default_location
    ep_id = parsed_ep.get("resource_id")
    ep_console_url = None
    if ep_id and ep_proj and ep_loc:
        ep_console_url = (
            f"https://console.cloud.google.com/vertex-ai/locations/{quote(ep_loc)}"
            f"/endpoints/{quote(ep_id)}?project={quote(ep_proj)}"
        )

    resources.append(
        {
            "category": "Vertex AI Endpoint",
            "label": ep_info.get("endpoint_display_name") or "Online Serving Endpoint",
            "resource_id": endpoint_res_name or "Not deployed yet",
            "status": deployer_stage.get("status", "PENDING"),
            "console_url": ep_console_url,
            "description": (
                f"Serving machine: {config_dict.get('deployment', {}).get('machine_type', 'N/A')} | "
                f"Replicas: {config_dict.get('deployment', {}).get('min_replica_count', 1)}.."
                f"{config_dict.get('deployment', {}).get('max_replica_count', 1)}"
            ),
        }
    )

    return resources


class TaskUIController:
    """Data access controller for reading task workspaces, datasets, logs, and evaluations."""

    def __init__(
        self,
        storage: StorageBackend | None = None,
        default_root_uri: str = "gs://distillfw-storage/tasks",
    ) -> None:
        self.storage = storage or StorageBackend()
        self.default_root_uri = default_root_uri

    def list_tasks(self, root_uri: str) -> dict[str, Any]:
        """Discover all distillation tasks under `root_uri`."""
        clean_root = root_uri.strip().rstrip("/")
        if not clean_root:
            raise ValueError("GCS root path cannot be empty.")

        uris = self.storage.list_uris(clean_root)
        state_uris = sorted(u for u in uris if u.endswith("/task_state.json"))

        tasks: list[dict[str, Any]] = []
        for state_uri in state_uris:
            try:
                raw = self.storage.read_json(state_uri)
                task_uri = raw.get("task_uri") or state_uri[: -len("/task_state.json")]
                tasks.append(
                    {
                        "task_id": raw.get("task_id") or task_uri.rsplit("/", 1)[-1],
                        "task_uri": task_uri,
                        "status": raw.get("status", "PENDING"),
                        "current_stage": raw.get("current_stage"),
                        "created_at": raw.get("created_at"),
                        "updated_at": raw.get("updated_at"),
                    }
                )
            except Exception as exc:
                task_uri = state_uri[: -len("/task_state.json")]
                tasks.append(
                    {
                        "task_id": task_uri.rsplit("/", 1)[-1],
                        "task_uri": task_uri,
                        "status": "UNKNOWN",
                        "current_stage": None,
                        "error": str(exc),
                    }
                )

        tasks.sort(key=lambda t: (t.get("updated_at") or "", t.get("task_id") or ""), reverse=True)
        return {
            "root_uri": clean_root,
            "count": len(tasks),
            "tasks": tasks,
        }

    def get_task_details(self, task_uri: str, refresh_vertex: bool = True) -> dict[str, Any]:
        """Load complete overview for a selected distillation task."""
        clean_task_uri = task_uri.strip().rstrip("/")
        ws = TaskWorkspace(task_uri=clean_task_uri, storage=self.storage)

        if refresh_vertex:
            try:
                state = ws.refresh_vertex_training_status()
            except Exception:
                state = ws.load_state()
        else:
            state = ws.load_state()

        state_dict = state.model_dump(mode="json")

        config_dict: dict[str, Any] = {}
        config_exists = self.storage.exists(ws.config_uri)
        if config_exists:
            try:
                config_text = self.storage.read_text(ws.config_uri)
                config_dict = yaml.safe_load(config_text) or {}
            except Exception:
                config_dict = {}

        project_id = (config_dict.get("gcp") or {}).get("project_id")

        # Discover log files under <task_uri>/logs/
        log_uris = sorted(
            [u for u in self.storage.list_uris(ws.logs_dir_uri) if u.endswith(".log")],
            reverse=True,
        )
        logs = [
            {
                "name": u.rsplit("/", 1)[-1],
                "uri": u,
                "console_url": _gcs_object_console_url(u, project_id=project_id),
            }
            for u in log_uris
        ]

        # Discover dataset files across stages
        datasets: list[dict[str, Any]] = []
        dataset_locations = [
            ("00_inputs", "Input Prompts", ws.inputs_dir_uri),
            ("01_raw_dataset", "Stage 1 — Raw Teacher Dataset", ws.raw_dataset_dir_uri),
            ("02_formatted_dataset", "Stage 2 — Formatted Split Dataset", ws.formatted_dataset_dir_uri),
            ("04_evaluation", "Stage 4 — Evaluation Predictions", ws.evaluation_dir_uri),
        ]
        for folder_key, stage_label, dir_uri in dataset_locations:
            for u in sorted(self.storage.list_uris(dir_uri)):
                fname = u.rsplit("/", 1)[-1]
                if fname.endswith((".parquet", ".jsonl", ".csv")):
                    datasets.append(
                        {
                            "name": fname,
                            "rel_path": f"{folder_key}/{fname}",
                            "stage_label": stage_label,
                            "format": fname.rsplit(".", 1)[-1].lower(),
                            "uri": u,
                            "console_url": _gcs_object_console_url(u, project_id=project_id),
                        }
                    )

        # Load evaluation scorecard if available (auto-repair any fallback-wrapped LLM-Judge JSON verdicts first)
        try:
            from distillfw.stages.evaluator import repair_evaluation_judge_artifacts

            if repair_evaluation_judge_artifacts(ws):
                state = ws.load_state()
                state_dict = state.model_dump(mode="json")
        except Exception:
            pass

        scorecard_uri = f"{ws.evaluation_dir_uri}/scorecard.json"
        scorecard: dict[str, Any] | None = None
        if self.storage.exists(scorecard_uri):
            try:
                scorecard = self.storage.read_json(scorecard_uri)
            except Exception:
                scorecard = None
        if scorecard is None:
            eval_stage = state_dict.get("stages", {}).get(StageName.MODEL_EVALUATOR.value, {})
            if isinstance(eval_stage.get("artifacts", {}).get("scorecard"), dict):
                scorecard = eval_stage["artifacts"]["scorecard"]

        predictions_uri = f"{ws.evaluation_dir_uri}/predictions.jsonl"
        has_predictions = self.storage.exists(predictions_uri)

        # Load endpoint_info.json if available
        endpoint_info_uri = f"{ws.deployment_dir_uri}/endpoint_info.json"
        endpoint_info: dict[str, Any] | None = None
        if self.storage.exists(endpoint_info_uri):
            try:
                endpoint_info = self.storage.read_json(endpoint_info_uri)
            except Exception:
                endpoint_info = None

        gcp_resources = _build_gcp_resources(
            ws=ws,
            state_dict=state_dict,
            config_dict=config_dict,
            endpoint_info=endpoint_info,
        )

        ordered_stages = []
        for st_name in ORDERED_STAGES:
            rec = state_dict.get("stages", {}).get(st_name.value, {})
            ordered_stages.append(
                {
                    "stage": st_name.value,
                    "status": rec.get("status", "PENDING"),
                    "started_at": rec.get("started_at"),
                    "completed_at": rec.get("completed_at"),
                    "attempt_count": rec.get("attempt_count", 0),
                    "error_message": rec.get("error_message"),
                    "progress_cursor": rec.get("progress_cursor", {}),
                    "artifacts": rec.get("artifacts", {}),
                }
            )

        return {
            "task_id": state.task_id,
            "task_uri": ws.task_uri,
            "status": state.status.value,
            "current_stage": state.current_stage.value if state.current_stage else None,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "config_sha256": state.config_sha256,
            "config_uri": ws.config_uri,
            "config_console_url": _gcs_object_console_url(ws.config_uri, project_id=project_id),
            "config_summary": {
                "description": config_dict.get("description", ""),
                "teacher_model": (config_dict.get("teacher") or {}).get("model_id", ""),
                "student_model": (config_dict.get("student") or {}).get("model_id", ""),
                "algorithm": (config_dict.get("training") or {}).get("algorithm", ""),
                "paradigm": (config_dict.get("training") or {}).get("paradigm", ""),
                "execution_mode": (config_dict.get("training") or {}).get("execution_mode", ""),
            },
            "stages": ordered_stages,
            "gcp_resources": gcp_resources,
            "logs": logs,
            "datasets": datasets,
            "scorecard": scorecard,
            "has_predictions": has_predictions,
            "predictions_uri": predictions_uri if has_predictions else None,
        }

    def get_task_config(self, task_uri: str) -> dict[str, Any]:
        """Return raw YAML and parsed dictionary for `<task_uri>/config.yaml`."""
        ws = TaskWorkspace(task_uri=task_uri.strip().rstrip("/"), storage=self.storage)
        yaml_text = self.storage.read_text(ws.config_uri)
        parsed = yaml.safe_load(yaml_text) or {}
        return {
            "task_uri": ws.task_uri,
            "config_uri": ws.config_uri,
            "yaml_text": yaml_text,
            "parsed": parsed,
        }

    def get_log_content(self, file_uri: str) -> dict[str, Any]:
        """Read a log file from GCS or local storage."""
        content = self.storage.read_text(file_uri)
        lines = content.splitlines()
        return {
            "file_uri": file_uri,
            "name": file_uri.rsplit("/", 1)[-1],
            "line_count": len(lines),
            "size_bytes": len(content.encode("utf-8")),
            "content": content,
        }

    def _load_dataframe(self, file_uri: str) -> pd.DataFrame:
        """Load a `.parquet`, `.jsonl`, or `.csv` file from `file_uri` into a DataFrame."""
        if file_uri.endswith(".parquet"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = self.storage.download_file(
                    file_uri, Path(tmp_dir) / "dataset.parquet"
                )
                return pd.read_parquet(local_path)
        if file_uri.endswith(".jsonl"):
            text = self.storage.read_text(file_uri)
            records = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
            return pd.DataFrame(records)
        if file_uri.endswith(".csv"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = self.storage.download_file(
                    file_uri, Path(tmp_dir) / "dataset.csv"
                )
                return pd.read_csv(local_path)
        raise ValueError(f"Unsupported dataset file extension for URI: {file_uri}")

    @staticmethod
    def _serialize_cell(val: Any) -> Any:
        """Convert pandas/numpy values into JSON-safe primitives or strings."""
        if val is None:
            return None
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=False)
        if hasattr(val, "tolist") and not isinstance(val, (str, bytes)):
            converted = val.tolist()
            if isinstance(converted, (list, dict)):
                return json.dumps(converted, ensure_ascii=False)
            return converted
        try:
            if pd.isna(val):
                return None
        except Exception:
            pass
        if isinstance(val, (int, float, bool, str)):
            return val
        return str(val)

    def get_dataset_table(
        self,
        file_uri: str,
        offset: int = 0,
        limit: int = 100,
        search_query: str = "",
    ) -> dict[str, Any]:
        """Read a dataset file and return structured columns and rows with search/pagination."""
        df = self._load_dataframe(file_uri)
        columns = list(df.columns)

        if search_query.strip():
            q = search_query.strip().lower()
            mask = df.apply(
                lambda row: any(
                    q in str(self._serialize_cell(v) or "").lower() for v in row
                ),
                axis=1,
            )
            df = df[mask]

        total_rows = int(len(df))
        safe_offset = max(0, offset)
        safe_limit = max(1, min(500, limit))
        sliced = df.iloc[safe_offset : safe_offset + safe_limit]

        rows: list[dict[str, Any]] = []
        for idx, (_, series) in enumerate(sliced.iterrows(), start=safe_offset + 1):
            row_dict = {"_row_num": idx}
            for col in columns:
                row_dict[col] = self._serialize_cell(series[col])
            rows.append(row_dict)

        return {
            "file_uri": file_uri,
            "name": file_uri.rsplit("/", 1)[-1],
            "columns": columns,
            "total_rows": total_rows,
            "offset": safe_offset,
            "limit": safe_limit,
            "rows": rows,
        }

    def get_evaluation_inferences(
        self,
        task_uri: str,
        split_filter: str = "all",
        search_query: str = "",
    ) -> dict[str, Any]:
        """Return side-by-side evaluation inferences (Before Training vs After Training vs Teacher)."""
        ws = TaskWorkspace(task_uri=task_uri.strip().rstrip("/"), storage=self.storage)
        predictions_uri = f"{ws.evaluation_dir_uri}/predictions.jsonl"
        if not self.storage.exists(predictions_uri):
            return {
                "task_uri": ws.task_uri,
                "predictions_uri": predictions_uri,
                "total_count": 0,
                "filtered_count": 0,
                "records": [],
            }

        from distillfw.stages.evaluator import (
            _load_workspace_tokenizer,
            compute_sample_lexical_metrics,
            count_prediction_tokens,
            parse_llm_judge_verdict,
            repair_evaluation_judge_artifacts,
        )

        try:
            repair_evaluation_judge_artifacts(ws)
        except Exception:
            pass

        workspace_tokenizer = _load_workspace_tokenizer(ws)
        raw_text = self.storage.read_text(predictions_uri)
        all_records: list[dict[str, Any]] = []
        for idx, line in enumerate(raw_text.splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            base_pred = item.get("base_student_prediction", "")
            distilled_pred = item.get("distilled_student_prediction", "")
            teacher_ref = item.get("teacher_reference", "")
            base_lat = item.get("base_latency_ms")
            distilled_lat = item.get("distilled_latency_ms")

            raw_base_m = item.get("base_metrics")
            if isinstance(raw_base_m, dict) and raw_base_m:
                base_metrics = dict(raw_base_m)
            else:
                base_metrics = compute_sample_lexical_metrics(base_pred, teacher_ref)
                if base_lat is not None:
                    base_metrics["latency_ms"] = round(float(base_lat), 2)

            raw_dist_m = item.get("distilled_metrics")
            if isinstance(raw_dist_m, dict) and raw_dist_m:
                distilled_metrics = dict(raw_dist_m)
            else:
                distilled_metrics = compute_sample_lexical_metrics(distilled_pred, teacher_ref)
                if distilled_lat is not None:
                    distilled_metrics["latency_ms"] = round(float(distilled_lat), 2)

            for m_obj, pred_text, lat_val in (
                (base_metrics, base_pred, base_lat),
                (distilled_metrics, distilled_pred, distilled_lat),
            ):
                if m_obj.get("output_tokens") is None or int(m_obj.get("output_tokens", 0)) <= 0:
                    m_obj["output_tokens"] = count_prediction_tokens(pred_text, workspace_tokenizer)
                effective_lat = m_obj.get("latency_ms", lat_val)
                if effective_lat is not None and m_obj.get("ms_per_output_token") is None:
                    try:
                        m_obj["ms_per_output_token"] = round(
                            float(effective_lat) / max(int(m_obj["output_tokens"]), 1), 2
                        )
                    except (TypeError, ValueError):
                        pass

                raw_reason = m_obj.get("llm_judge_reason")
                if isinstance(raw_reason, str) and (
                    "```" in raw_reason or '"score"' in raw_reason
                ):
                    parsed = parse_llm_judge_verdict(raw_reason)
                    if parsed.get("_parsed_ok"):
                        m_obj["llm_judge_score"] = parsed["score"]
                        m_obj["llm_judge_reason"] = parsed["reason"]

            metrics_delta: dict[str, float] = {}
            for m_key in ("exact_match", "rouge1", "rouge2", "bleu", "llm_judge_score"):
                if m_key in base_metrics and m_key in distilled_metrics:
                    try:
                        metrics_delta[m_key] = round(
                            float(distilled_metrics[m_key]) - float(base_metrics[m_key]), 4
                        )
                    except (TypeError, ValueError):
                        pass
            for sys_key in ("output_tokens", "ms_per_output_token", "latency_ms"):
                if sys_key in base_metrics and sys_key in distilled_metrics:
                    try:
                        metrics_delta[sys_key] = round(
                            float(distilled_metrics[sys_key]) - float(base_metrics[sys_key]), 2
                        )
                    except (TypeError, ValueError):
                        pass

            all_records.append(
                {
                    "index": idx,
                    "split": item.get("split", "test"),
                    "prompt": item.get("prompt", ""),
                    "student_before_training": base_pred,
                    "student_after_training": distilled_pred,
                    "teacher_model": teacher_ref,
                    "base_latency_ms": base_lat,
                    "distilled_latency_ms": distilled_lat,
                    "base_metrics": base_metrics,
                    "distilled_metrics": distilled_metrics,
                    "metrics_delta": metrics_delta,
                }
            )

        filtered = all_records
        if split_filter and split_filter.lower() in ("test", "train"):
            target_split = split_filter.lower()
            filtered = [r for r in filtered if str(r.get("split", "")).lower() == target_split]

        if search_query.strip():
            q = search_query.strip().lower()
            filtered = [
                r
                for r in filtered
                if q in str(r.get("prompt", "")).lower()
                or q in str(r.get("student_before_training", "")).lower()
                or q in str(r.get("student_after_training", "")).lower()
                or q in str(r.get("teacher_model", "")).lower()
            ]

        return {
            "task_uri": ws.task_uri,
            "predictions_uri": predictions_uri,
            "total_count": len(all_records),
            "filtered_count": len(filtered),
            "records": filtered,
        }


def create_ui_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    controller: TaskUIController | None = None,
) -> ThreadingHTTPServer:
    """Create a configured `ThreadingHTTPServer` serving the `distillfw` Web UI and JSON API."""
    ui_controller = controller or TaskUIController()

    class DistillFWUIHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_static(self, file_path: Path) -> None:
            if not file_path.exists() or not file_path.is_file():
                self._send_json({"error": "File not found"}, status=HTTPStatus.NOT_FOUND)
                return
            mime_type, _ = mimetypes.guess_type(str(file_path))
            content = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{mime_type or 'application/octet-stream'}; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:  # noqa: N802
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            params = {k: v[0] for k, v in parse_qs(parsed_url.query).items()}

            try:
                if path == "/api/defaults":
                    self._send_json({"default_root_uri": ui_controller.default_root_uri})
                    return

                if path == "/api/tasks":
                    root_uri = params.get("root_uri", ui_controller.default_root_uri)
                    self._send_json(ui_controller.list_tasks(root_uri))
                    return

                if path == "/api/task/details":
                    task_uri = params.get("task_uri", "")
                    if not task_uri:
                        self._send_json({"error": "Missing 'task_uri' parameter."}, status=HTTPStatus.BAD_REQUEST)
                        return
                    refresh = params.get("refresh", "true").lower() != "false"
                    self._send_json(ui_controller.get_task_details(task_uri, refresh_vertex=refresh))
                    return

                if path == "/api/task/config":
                    task_uri = params.get("task_uri", "")
                    if not task_uri:
                        self._send_json({"error": "Missing 'task_uri' parameter."}, status=HTTPStatus.BAD_REQUEST)
                        return
                    self._send_json(ui_controller.get_task_config(task_uri))
                    return

                if path == "/api/task/log":
                    file_uri = params.get("file_uri", "")
                    if not file_uri:
                        self._send_json({"error": "Missing 'file_uri' parameter."}, status=HTTPStatus.BAD_REQUEST)
                        return
                    self._send_json(ui_controller.get_log_content(file_uri))
                    return

                if path == "/api/task/dataset":
                    file_uri = params.get("file_uri", "")
                    if not file_uri:
                        self._send_json({"error": "Missing 'file_uri' parameter."}, status=HTTPStatus.BAD_REQUEST)
                        return
                    offset = int(params.get("offset", "0"))
                    limit = int(params.get("limit", "100"))
                    search_q = params.get("q", "")
                    self._send_json(
                        ui_controller.get_dataset_table(
                            file_uri=file_uri,
                            offset=offset,
                            limit=limit,
                            search_query=search_q,
                        )
                    )
                    return

                if path == "/api/task/inferences":
                    task_uri = params.get("task_uri", "")
                    if not task_uri:
                        self._send_json({"error": "Missing 'task_uri' parameter."}, status=HTTPStatus.BAD_REQUEST)
                        return
                    split_filter = params.get("split", "all")
                    search_q = params.get("q", "")
                    self._send_json(
                        ui_controller.get_evaluation_inferences(
                            task_uri=task_uri,
                            split_filter=split_filter,
                            search_query=search_q,
                        )
                    )
                    return

                # Static files
                if path in ("/", "/index.html"):
                    self._send_static(STATIC_DIR / "index.html")
                    return

                rel = path.lstrip("/")
                candidate = (STATIC_DIR / rel).resolve()
                if STATIC_DIR.resolve() in candidate.parents and candidate.is_file():
                    self._send_static(candidate)
                    return

                self._send_json({"error": f"Not found: {path}"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json(
                    {"error": str(exc) or exc.__class__.__name__},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Silence noisy request logs in CLI / tests
            return

    return ThreadingHTTPServer((host, port), DistillFWUIHandler)


def run_ui_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    default_root_uri: str = "gs://distillfw-storage/tasks",
    storage: StorageBackend | None = None,
) -> None:
    """Start the `distillfw` Web UI HTTP server and block until interrupted."""
    controller = TaskUIController(storage=storage, default_root_uri=default_root_uri)
    server = create_ui_server(host=host, port=port, controller=controller)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
