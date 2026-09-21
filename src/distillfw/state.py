"""Self-contained GCS task state manifest and stateless resumption manager."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
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

    @classmethod
    def initialize(
        cls,
        config: DistillationConfig,
        prompts_path: str | Path,
        storage: StorageBackend | None = None,
    ) -> TaskWorkspace:
        """Initialize a new self-contained task workspace on GCS."""
        backend = storage or StorageBackend(project_id=config.gcp.project_id)
        ws = cls(task_uri=config.task_uri, storage=backend)

        # Persist frozen config snapshot
        ws.storage.write_text(ws.config_uri, config.to_yaml())

        # Persist immutable copy of input prompts
        prompts_src = Path(prompts_path)
        prompts_dest_uri = f"{ws.inputs_dir_uri}/{prompts_src.name}"
        ws.storage.upload_file(prompts_src, prompts_dest_uri)

        # Initialize and persist task_state.json
        state = TaskState(
            task_id=config.task_id,
            task_uri=ws.task_uri,
            config_sha256=config.sha256(),
        )
        state.stages[StageName.DATASET_GENERATOR].artifacts["input_prompts_uri"] = prompts_dest_uri
        ws.save_state(state)
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
        return state

    def update_stage_cursor(self, stage: StageName, cursor_updates: dict[str, Any]) -> TaskState:
        """Update intra-stage progress pointers on GCS for stateless resumption."""
        state = self.load_state()
        state.stages[stage].progress_cursor.update(cursor_updates)
        self.save_state(state)
        return state

    def mark_stage_completed(
        self, stage: StageName, artifacts: dict[str, Any] | None = None
    ) -> TaskState:
        """Transition a stage to COMPLETED and record produced GCS artifacts."""
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
        return state

    def mark_stage_failed(self, stage: StageName, error_message: str) -> TaskState:
        """Transition a stage to FAILED and persist error context to GCS."""
        state = self.load_state()
        rec = state.stages[stage]
        rec.status = StageStatus.FAILED
        rec.error_message = error_message
        state.status = StageStatus.FAILED
        self.save_state(state)
        return state

    def next_pending_stage(self) -> StageName | None:
        """Return the first stage that is not COMPLETED (for stateless pipeline continuation)."""
        state = self.load_state()
        for stage in ORDERED_STAGES:
            if state.stages[stage].status != StageStatus.COMPLETED:
                return stage
        return None
