"""Self-contained GCS task state manifest and stateless resumption manager."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from distillfw.config import DistillationConfig
from distillfw.gcp.storage import StorageBackend


class StageName(str, Enum):
    """Canonical ordered stages of the distillation pipeline."""

    DATASET_GENERATOR = "dataset_generator"
    DATASET_FORMATTER = "dataset_formatter"
    MODEL_TRAINER = "model_trainer"
    MODEL_EVALUATOR = "model_evaluator"
    MODEL_DEPLOYER = "model_deployer"


ORDERED_STAGES: list[StageName] = [
    StageName.DATASET_GENERATOR,
    StageName.DATASET_FORMATTER,
    StageName.MODEL_TRAINER,
    StageName.MODEL_EVALUATOR,
    StageName.MODEL_DEPLOYER,
]


class StageStatus(str, Enum):
    """Lifecycle status of an individual stage or overall task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class StageRecord(BaseModel):
    """Per-stage execution metadata persisted in `task_state.json`."""

    stage: StageName
    status: StageStatus = Field(default=StageStatus.PENDING)
    started_at: str | None = Field(default=None)
    completed_at: str | None = Field(default=None)
    attempt_count: int = Field(default=0)
    error_message: str | None = Field(default=None)
    progress_cursor: dict[str, Any] = Field(
        default_factory=dict,
        description="Intra-stage resumption pointers (shard index, step checkpoint, Vertex job ID)",
    )
    artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="Explicit GCS URIs of inputs consumed and outputs produced by this stage",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskState(BaseModel):
    """Atomic GCS state manifest (`task_state.json`) for a distillation task."""

    task_id: str
    task_uri: str
    config_sha256: str
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)
    status: StageStatus = Field(default=StageStatus.PENDING)
    current_stage: StageName | None = Field(default=None)
    stages: dict[StageName, StageRecord] = Field(
        default_factory=lambda: {
            stage: StageRecord(stage=stage) for stage in ORDERED_STAGES
        }
    )


class TaskInitializationError(ValueError):
    """Raised when task initialization fails preflight configuration or teacher model checks."""


class TaskWorkspaceExistsError(TaskInitializationError):
    """Raised when attempting to initialize a task workspace that already has contents on GCS."""


class TaskResetAbortedError(TaskInitializationError):
    """Raised when the user declines confirmation to erase existing GCS workspace contents."""


def verify_hf_token_present() -> str:
    """Verify that the `HF_TOKEN` environment variable is set and non-empty.

    Raises `TaskInitializationError` with actionable instructions on how to create
    and export a Hugging Face access token if `HF_TOKEN` is unset or empty.
    """
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise TaskInitializationError(
            "Error: The HF_TOKEN environment variable is not set or is empty.\n"
            "A valid Hugging Face access token is required to download gated Gemma student models "
            "during training and evaluation.\n\n"
            "How to fix this:\n"
            "  1. Create or copy a Hugging Face access token at:\n"
            "     https://huggingface.co/settings/tokens\n"
            "  2. Ensure your Hugging Face account has accepted the Gemma license agreement.\n"
            "  3. Export HF_TOKEN in your shell before running distillfw:\n"
            "     export HF_TOKEN=\"hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\""
        )
    return token


def verify_gcp_bucket_location(
    config: DistillationConfig,
    storage: StorageBackend,
) -> None:
    """Verify that the configured GCS bucket(s) exist and match `config.gcp.location`.

    Vertex AI Custom Training Jobs require single-region GCS buckets in the exact same
    region as `gcp.location` (e.g., `us-central1`). Multi-region buckets (such as `us`,
    `eu`, or `asia`) or buckets in a different region fail at training submission time
    with `400 FailedPrecondition`.
    """
    from distillfw.gcp.storage import parse_gcs_uri

    if config.gcp is None:
        return

    expected_location = config.gcp.location.strip().lower()
    project_id = config.gcp.project_id
    buckets_to_check: list[tuple[str, str]] = [
        ("gcp.bucket_name", config.gcp.bucket_name),
    ]
    if config.gcp.staging_bucket:
        staging_bucket_name, _ = parse_gcs_uri(config.gcp.staging_bucket)
        if staging_bucket_name != config.gcp.bucket_name:
            buckets_to_check.append(("gcp.staging_bucket", staging_bucket_name))

    for field_name, bucket_name in buckets_to_check:
        try:
            actual_location = storage.get_bucket_location(bucket_name)
        except FileNotFoundError as exc:
            raise TaskInitializationError(
                f"Initialization error: Cloud Storage bucket 'gs://{bucket_name}' (configured in {field_name}) "
                f"does not exist or is not accessible in project '{project_id}'.\n\n"
                f"Create the bucket in region '{config.gcp.location}' using:\n"
                f"  gcloud storage buckets create gs://{bucket_name} \\\n"
                f"    --project={project_id} \\\n"
                f"    --location={config.gcp.location} \\\n"
                f"    --uniform-bucket-level-access\n\n"
                f"And ensure config.yaml has:\n"
                f"  gcp:\n"
                f"    project_id: {project_id}\n"
                f"    location: {config.gcp.location}\n"
                f"    bucket_name: {bucket_name}"
            ) from exc

        if actual_location is None:
            # Local emulation backend; skip live region check
            continue

        actual_loc_lower = actual_location.strip().lower()
        if actual_loc_lower != expected_location:
            suggested_bucket = f"{bucket_name}-{expected_location}"
            raise TaskInitializationError(
                f"Initialization error: Cloud Storage bucket 'gs://{bucket_name}' (configured in {field_name}) "
                f"is in location '{actual_loc_lower}', which does not match gcp.location='{config.gcp.location}'.\n"
                f"Vertex AI Custom Training Jobs and Model Registry require a single-region GCS bucket in the exact "
                f"same region as gcp.location ('{config.gcp.location}'). Multi-region buckets (such as 'us', 'eu', "
                f"'asia') or cross-region buckets are rejected by Vertex AI.\n\n"
                f"How to fix this:\n"
                f"  Option A (Create a new regional bucket in '{config.gcp.location}'):\n"
                f"    gcloud storage buckets create gs://{suggested_bucket} \\\n"
                f"      --project={project_id} \\\n"
                f"      --location={config.gcp.location} \\\n"
                f"      --uniform-bucket-level-access\n\n"
                f"    Then update your config.yaml:\n"
                f"      gcp:\n"
                f"        project_id: {project_id}\n"
                f"        location: {config.gcp.location}\n"
                f"        bucket_name: {suggested_bucket}\n\n"
                f"  Option B (Recreate 'gs://{bucket_name}' in '{config.gcp.location}' if it can be replaced):\n"
                f"    gcloud storage rm --recursive gs://{bucket_name}/**\n"
                f"    gcloud storage buckets delete gs://{bucket_name} --project={project_id}\n"
                f"    gcloud storage buckets create gs://{bucket_name} \\\n"
                f"      --project={project_id} \\\n"
                f"      --location={config.gcp.location} \\\n"
                f"      --uniform-bucket-level-access"
            )


def verify_teacher_logprobs_support(
    config: DistillationConfig,
    teacher_callable: Any = None,
) -> None:
    """Run preflight checks verifying logprobs configuration compatibility and live teacher support.

    1) Verifies that `teacher.response_logprobs` and `teacher.logprobs_top_k` match the
       requirements of `training.algorithm`.
    2) If `teacher.response_logprobs` is True, executes a test inference against the teacher
       model to verify that logprobs are supported and that at least `logprobs_top_k` candidates
       are delivered per token step.
    """
    try:
        config.validate_logprobs_compatibility()
    except ValueError as exc:
        raise TaskInitializationError(str(exc)) from exc

    if not config.teacher.response_logprobs:
        return

    assert config.teacher.logprobs_top_k is not None
    required_top_k = config.teacher.logprobs_top_k

    if teacher_callable is not None:
        try:
            probe_res = teacher_callable("Preflight logprobs verification probe.", {"probe": True})
        except Exception as exc:
            raise TaskInitializationError(
                f"Initialization error: Teacher model '{config.teacher.model_id}' failed "
                f"preflight logprobs test inference: {exc}"
            ) from exc
        topk_steps = probe_res.get("topk_logprobs")
    else:
        from google import genai
        from google.genai import types

        from distillfw.gcp.retry import call_with_exponential_backoff

        default_location = config.gcp.location if config.gcp is not None else "us-central1"
        teacher_location = config.teacher.location or default_location
        project_id = config.gcp.project_id if config.gcp is not None else None
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=teacher_location,
        )
        try:
            response = call_with_exponential_backoff(
                lambda: client.chats.create(
                    model=config.teacher.model_id,
                    config=types.GenerateContentConfig(
                        max_output_tokens=16,
                        temperature=config.teacher.temperature,
                        response_logprobs=True,
                        logprobs=required_top_k,
                    ),
                ).send_message(message="Preflight logprobs verification probe."),
                max_retries=config.teacher.max_retries,
                initial_delay=config.teacher.initial_retry_delay_seconds,
                max_delay=config.teacher.max_retry_delay_seconds,
                multiplier=config.teacher.backoff_multiplier,
                operation_name=f"Preflight logprobs probe ({config.teacher.model_id})",
            )
        except Exception as exc:
            raise TaskInitializationError(
                f"Initialization error: Teacher model '{config.teacher.model_id}' "
                f"(location='{teacher_location}') failed preflight logprobs test inference: {exc}"
            ) from exc

        topk_steps = []
        if getattr(response, "candidates", None):
            first_cand = response.candidates[0]
            if getattr(first_cand, "logprobs_result", None):
                for step in getattr(first_cand.logprobs_result, "top_candidates", []) or []:
                    step_candidates = [
                        {"token": c.token, "log_probability": c.log_probability}
                        for c in getattr(step, "candidates", [])
                    ]
                    topk_steps.append({"top_candidates": step_candidates})

    if not topk_steps:
        raise TaskInitializationError(
            f"Initialization error: Teacher model '{config.teacher.model_id}' did not return "
            f"any token logprobs during preflight test inference."
        )

    for step_idx, step_obj in enumerate(topk_steps):
        cands = step_obj.get("top_candidates", [])
        if len(cands) < required_top_k:
            raise TaskInitializationError(
                f"Initialization error: Teacher model '{config.teacher.model_id}' returned only "
                f"{len(cands)} logprob candidates at step {step_idx}, fewer than required "
                f"teacher.logprobs_top_k={required_top_k}."
            )


def verify_judge_model_access(
    config: DistillationConfig,
    storage: StorageBackend | None = None,
    judge_callable: Any = None,
) -> None:
    """Verify that the configured evaluation judge model (`judge_model_id` in `judge_model_location`) is accessible.

    Executed during `distillfw init` preflight checks and at the start of Stage 4 (`model_evaluator`).
    If `llm_judge` is included in `config.evaluation.metrics` and the judge model cannot be accessed,
    raises `TaskInitializationError` with actionable remediation steps.
    """
    from distillfw.config import EvaluationMetric

    if EvaluationMetric.LLM_JUDGE not in config.evaluation.metrics:
        return

    judge_model_id = config.evaluation.judge_model_id
    default_loc = config.gcp.location if config.gcp is not None else "us-central1"
    judge_location = (
        config.evaluation.resolved_judge_location
        or config.teacher.location
        or default_loc
    )
    project_id = config.gcp.project_id if config.gcp is not None else "local"

    def _build_error_message(exc: Exception) -> str:
        return (
            f"Judge model access error: Cannot access evaluation judge model '{judge_model_id}' "
            f"in location '{judge_location}' (project '{project_id}'):\n{exc}\n\n"
            f"How to fix this:\n"
            f"  1. Update the 'evaluation' section in your config.yaml to specify a valid judge_model_id "
            f"and judge_model_location that your project has access to (e.g., matching your teacher model):\n"
            f"     evaluation:\n"
            f"       judge_model_id: {config.teacher.model_id}\n"
            f"       judge_model_location: {config.teacher.location or default_loc}\n"
            f"  2. If your task is already initialized on GCS and you want to apply the updated config.yaml "
            f"and reset the evaluation stage without re-running training, run:\n"
            f"     distillfw reset-eval {config.task_uri} --config <path/to/config.yaml>"
        )

    if judge_callable is not None:
        try:
            judge_callable("Preflight judge access probe.", "Reference", "Prediction")
        except Exception as exc:
            raise TaskInitializationError(_build_error_message(exc)) from exc
        return

    if config.gcp is None or (storage is not None and storage.is_local_emulation):
        return

    try:
        from google import genai
        from google.genai import types

        from distillfw.gcp.retry import call_with_exponential_backoff

        client = genai.Client(
            vertexai=True,
            project=config.gcp.project_id,
            location=judge_location,
        )
        gen_cfg = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8,
        )
        call_with_exponential_backoff(
            lambda: client.chats.create(model=judge_model_id, config=gen_cfg).send_message(
                message="Reply OK."
            ),
            max_retries=config.teacher.max_retries,
            initial_delay=config.teacher.initial_retry_delay_seconds,
            max_delay=config.teacher.max_retry_delay_seconds,
            multiplier=config.teacher.backoff_multiplier,
            operation_name=f"Preflight judge access probe ({judge_model_id} @ {judge_location})",
        )
    except Exception as exc:
        raise TaskInitializationError(_build_error_message(exc)) from exc


class TaskWorkspace:
    """Stateless manager for an isolated distillation task workspace on GCS.

    Every path and operation is derived strictly from `task_uri` (`gs://<bucket>/tasks/<task_id>`).
    """

    def __init__(self, task_uri: str, storage: StorageBackend | None = None) -> None:
        self.task_uri = task_uri.rstrip("/")
        self.storage = storage or StorageBackend()

    # Standard GCS directory & file layout
    @property
    def config_uri(self) -> str:
        return f"{self.task_uri}/config.yaml"

    @property
    def state_uri(self) -> str:
        return f"{self.task_uri}/task_state.json"

    @property
    def inputs_dir_uri(self) -> str:
        return f"{self.task_uri}/00_inputs"

    @property
    def raw_dataset_dir_uri(self) -> str:
        return f"{self.task_uri}/01_raw_dataset"

    @property
    def formatted_dataset_dir_uri(self) -> str:
        return f"{self.task_uri}/02_formatted_dataset"

    @property
    def checkpoints_dir_uri(self) -> str:
        return f"{self.task_uri}/03_checkpoints"

    @property
    def evaluation_dir_uri(self) -> str:
        return f"{self.task_uri}/04_evaluation"

    @property
    def exported_model_dir_uri(self) -> str:
        return f"{self.task_uri}/05_exported_model"

    @property
    def deployment_dir_uri(self) -> str:
        return f"{self.task_uri}/06_deployment"

    @property
    def logs_dir_uri(self) -> str:
        return f"{self.task_uri}/logs"

    @classmethod
    def initialize(
        cls,
        config: DistillationConfig,
        prompts_path: str | Path,
        storage: StorageBackend | None = None,
        teacher_callable: Any = None,
        judge_callable: Any = None,
        force_reset: bool = False,
        confirm_reset_callback: Any = None,
    ) -> TaskWorkspace:
        """Initialize a new self-contained task workspace on GCS after preflight validation.

        If existing contents are found under `config.task_uri`:
        - When `force_reset` is False, raises `TaskWorkspaceExistsError` without modifying anything.
        - When `force_reset` is True, asks for confirmation (if `confirm_reset_callback` is provided)
          and completely erases all existing contents under `config.task_uri` before proceeding.
        """
        from distillfw.logging_utils import get_logger

        logger = get_logger("state")
        verify_hf_token_present()
        project_id = config.gcp.project_id if config.gcp is not None else None
        backend = storage or StorageBackend(project_id=project_id)
        verify_gcp_bucket_location(config, backend)
        ws = cls(task_uri=config.task_uri, storage=backend)

        existing_uris = ws.storage.list_uris(ws.task_uri)
        if existing_uris and not force_reset:
            logger.warning(
                "Target GCS workspace '%s' already contains %d existing object(s); aborting init.",
                ws.task_uri,
                len(existing_uris),
            )
            raise TaskWorkspaceExistsError(
                f"Target GCS workspace '{ws.task_uri}' already contains {len(existing_uris)} "
                f"existing object(s). Exiting without modifying anything. "
                f"Pass --force-reset to wipe and re-initialize this workspace."
            )

        if existing_uris and force_reset:
            if confirm_reset_callback is not None:
                confirmed = confirm_reset_callback(ws.task_uri, existing_uris)
                if not confirmed:
                    logger.warning(
                        "User declined confirmation to erase %d existing object(s) in %s; aborting.",
                        len(existing_uris),
                        ws.task_uri,
                    )
                    raise TaskResetAbortedError(
                        f"Aborted --force-reset for '{ws.task_uri}'. Existing contents were left untouched."
                    )

            deleted = ws.storage.delete_prefix(ws.task_uri)
            logger.info(
                "Force-reset confirmed: completely erased all %d existing object(s) under %s before proceeding.",
                deleted,
                ws.task_uri,
            )

        logger.info(
            "Running preflight validation for task '%s' (algorithm=%s, teacher=%s, response_logprobs=%s, judge=%s)...",
            config.task_id,
            config.training.algorithm.value,
            config.teacher.model_id,
            config.teacher.response_logprobs,
            config.evaluation.judge_model_id,
        )
        verify_teacher_logprobs_support(config, teacher_callable=teacher_callable)
        verify_judge_model_access(config, storage=backend, judge_callable=judge_callable)
        logger.info("Preflight validation succeeded for task '%s'.", config.task_id)

        # Persist frozen config snapshot
        ws.storage.write_text(ws.config_uri, config.to_yaml())
        logger.info("Uploaded frozen configuration snapshot to %s", ws.config_uri)

        # Persist immutable copy of input prompts
        prompts_src = Path(prompts_path)
        prompts_dest_uri = f"{ws.inputs_dir_uri}/{prompts_src.name}"
        ws.storage.upload_file(prompts_src, prompts_dest_uri)
        logger.info("Uploaded input prompt dataset to %s", prompts_dest_uri)

        # Initialize and persist task_state.json
        state = TaskState(
            task_id=config.task_id,
            task_uri=ws.task_uri,
            config_sha256=config.sha256(),
        )
        state.stages[StageName.DATASET_GENERATOR].artifacts["input_prompts_uri"] = prompts_dest_uri
        ws.save_state(state)
        logger.info("Initialized task state manifest at %s", ws.state_uri)
        return ws

    def load_config(self) -> DistillationConfig:
        """Load the frozen `DistillationConfig` directly from `<task_uri>/config.yaml`."""
        yaml_text = self.storage.read_text(self.config_uri)
        return DistillationConfig.from_yaml(yaml_text)

    def load_state(self) -> TaskState:
        """Load `TaskState` directly from `<task_uri>/task_state.json`."""
        raw = self.storage.read_json(self.state_uri)
        return TaskState.model_validate(raw)

    def save_state(self, state: TaskState) -> None:
        """Atomically save `TaskState` to `<task_uri>/task_state.json`."""
        state.updated_at = _utc_now()
        self.storage.write_json(self.state_uri, state.model_dump(mode="json"))

    def mark_stage_running(
        self, stage: StageName, cursor_updates: dict[str, Any] | None = None
    ) -> TaskState:
        """Transition a stage to RUNNING and persist state to GCS."""
        from distillfw.logging_utils import get_logger

        state = self.load_state()
        rec = state.stages[stage]
        rec.status = StageStatus.RUNNING
        rec.started_at = rec.started_at or _utc_now()
        rec.attempt_count += 1
        rec.error_message = None
        if cursor_updates:
            rec.progress_cursor.update(cursor_updates)
        state.current_stage = stage
        state.status = StageStatus.RUNNING
        self.save_state(state)
        get_logger("state").info(
            "Stage [%s] -> RUNNING (attempt #%d, cursor=%s)",
            stage.value,
            rec.attempt_count,
            rec.progress_cursor,
        )
        return state

    def update_stage_cursor(self, stage: StageName, cursor_updates: dict[str, Any]) -> TaskState:
        """Update intra-stage progress pointers on GCS for stateless resumption."""
        from distillfw.logging_utils import get_logger

        state = self.load_state()
        state.stages[stage].progress_cursor.update(cursor_updates)
        self.save_state(state)
        get_logger("state").info(
            "Stage [%s] progress cursor updated: %s",
            stage.value,
            state.stages[stage].progress_cursor,
        )
        return state

    def mark_stage_completed(
        self, stage: StageName, artifacts: dict[str, Any] | None = None
    ) -> TaskState:
        """Transition a stage to COMPLETED and record produced GCS artifacts."""
        from distillfw.logging_utils import get_logger

        state = self.load_state()
        rec = state.stages[stage]
        rec.status = StageStatus.COMPLETED
        rec.completed_at = _utc_now()
        if artifacts:
            rec.artifacts.update(artifacts)
        # Check if all stages are complete
        if all(state.stages[s].status == StageStatus.COMPLETED for s in ORDERED_STAGES):
            state.status = StageStatus.COMPLETED
            state.current_stage = None
        self.save_state(state)
        get_logger("state").info(
            "Stage [%s] -> COMPLETED | overall task status: %s",
            stage.value,
            state.status.value,
        )
        return state

    def mark_stage_failed(self, stage: StageName, error_message: str) -> TaskState:
        """Transition a stage to FAILED and persist error context to GCS."""
        from distillfw.logging_utils import get_logger

        state = self.load_state()
        rec = state.stages[stage]
        rec.status = StageStatus.FAILED
        rec.error_message = error_message
        state.status = StageStatus.FAILED
        self.save_state(state)
        get_logger("state").error(
            "Stage [%s] -> FAILED: %s",
            stage.value,
            error_message,
        )
        return state

    def reset_stage(
        self, stage: StageName, new_config: DistillationConfig | None = None
    ) -> TaskState:
        """Reset a stage back to PENDING (clearing cursor, attempts, errors, and timestamps) and persist to GCS."""
        from distillfw.logging_utils import get_logger

        logger = get_logger("state")
        state = self.load_state()
        if new_config is not None:
            self.storage.write_text(self.config_uri, new_config.to_yaml())
            state.config_sha256 = new_config.sha256()
            logger.info("Updated frozen configuration snapshot at %s", self.config_uri)

        state.stages[stage] = StageRecord(stage=stage)

        if state.current_stage == stage:
            state.current_stage = None

        # Recompute overall task status across all stages
        stage_statuses = [state.stages[s].status for s in ORDERED_STAGES]
        if any(s == StageStatus.FAILED for s in stage_statuses):
            state.status = StageStatus.FAILED
        elif any(s == StageStatus.RUNNING for s in stage_statuses):
            state.status = StageStatus.RUNNING
        elif all(s == StageStatus.COMPLETED for s in stage_statuses):
            state.status = StageStatus.COMPLETED
        else:
            state.status = StageStatus.PENDING

        self.save_state(state)
        logger.info(
            "Stage [%s] reset to PENDING | overall task status: %s",
            stage.value,
            state.status.value,
        )
        return state

    def next_pending_stage(self) -> StageName | None:
        """Return the first stage that is not COMPLETED (for stateless pipeline continuation)."""
        state = self.load_state()
        for stage in ORDERED_STAGES:
            if state.stages[stage].status != StageStatus.COMPLETED:
                return stage
        return None

    def verify_upstream_stages_completed(self, stage: StageName) -> None:
        """Ensure every upstream stage preceding `stage` in ORDERED_STAGES has status COMPLETED."""
        state = self.load_state()
        stage_idx = ORDERED_STAGES.index(stage)
        for upstream_stage in ORDERED_STAGES[:stage_idx]:
            upstream_rec = state.stages[upstream_stage]
            if upstream_rec.status != StageStatus.COMPLETED:
                raise RuntimeError(
                    f"Cannot execute stage '{stage.value}' because upstream prerequisite stage "
                    f"'{upstream_stage.value}' is currently {upstream_rec.status.value} "
                    f"(must be COMPLETED before '{stage.value}' can run)."
                )

    def refresh_vertex_training_status(self) -> TaskState:
        """If `model_trainer` is RUNNING with a Vertex AI Custom Job, query its live status
        on Vertex AI, update the stage/task status and progress cursor accordingly,
        persist the updated `task_state.json` to GCS, and return the refreshed `TaskState`.
        """
        state = self.load_state()
        trainer_rec = state.stages[StageName.MODEL_TRAINER]
        job_resource_name = trainer_rec.progress_cursor.get("vertex_job_resource_name")

        if trainer_rec.status != StageStatus.RUNNING or not job_resource_name:
            return state

        config = self.load_config()
        if config.gcp is None:
            return state

        from distillfw.gcp.vertex import VertexJobManager, normalize_vertex_job_state

        job_mgr = VertexJobManager(config.gcp)
        job_info = job_mgr.get_job_status(job_resource_name)
        normalized_state = normalize_vertex_job_state(job_info.get("state", ""))

        trainer_rec.progress_cursor["vertex_job_state"] = normalized_state

        terminal_success = {"JOB_STATE_SUCCEEDED"}
        terminal_failure = {
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_CANCELLING",
            "JOB_STATE_EXPIRED",
        }

        if normalized_state in terminal_success:
            trainer_rec.status = StageStatus.COMPLETED
            trainer_rec.completed_at = trainer_rec.completed_at or _utc_now()
            trainer_rec.error_message = None
            trainer_rec.artifacts.update(
                {
                    "exported_model_uri": self.exported_model_dir_uri,
                    "vertex_job_resource_name": job_resource_name,
                }
            )
            if all(state.stages[s].status == StageStatus.COMPLETED for s in ORDERED_STAGES):
                state.status = StageStatus.COMPLETED
                state.current_stage = None
            elif any(state.stages[s].status == StageStatus.FAILED for s in ORDERED_STAGES):
                state.status = StageStatus.FAILED
            else:
                state.status = StageStatus.RUNNING
        elif normalized_state in terminal_failure:
            err_detail = job_info.get("error") or f"Vertex AI job terminated with state {normalized_state}"
            trainer_rec.status = StageStatus.FAILED
            trainer_rec.error_message = err_detail
            state.status = StageStatus.FAILED
        else:
            state.status = StageStatus.RUNNING

        self.save_state(state)
        return state
