"""Stateless pipeline orchestrator driven solely by the GCS task workspace URI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from distillfw.config import DistillationConfig
from distillfw.gcp.storage import StorageBackend
from distillfw.stages.deployer import ModelDeployer
from distillfw.stages.evaluator import ModelEvaluator
from distillfw.stages.formatter import DatasetFormatter
from distillfw.stages.generator import DatasetGenerator
from distillfw.stages.trainer import ModelTrainer
from distillfw.state import ORDERED_STAGES, StageName, StageStatus, TaskState, TaskWorkspace


class DistillationPipeline:
    """Stateless 5-stage distillation pipeline orchestrator.

    Can be initialized on any machine or container using ONLY `task_uri`:
        pipeline = DistillationPipeline.from_task_uri("gs://my-bucket/tasks/task-001")
        pipeline.resume()
    """

    def __init__(
        self,
        workspace: TaskWorkspace,
        teacher_callable: Any = None,
        tokenizer: Any = None,
        custom_train_fn: Any = None,
        student_predict_fn: Any = None,
        base_student_predict_fn: Any = None,
        judge_fn: Any = None,
        deploy_fn: Any = None,
    ) -> None:
        self.workspace = workspace
        self.teacher_callable = teacher_callable
        self.tokenizer = tokenizer
        self.custom_train_fn = custom_train_fn
        self.student_predict_fn = student_predict_fn
        self.base_student_predict_fn = base_student_predict_fn
        self.judge_fn = judge_fn
        self.deploy_fn = deploy_fn

    @classmethod
    def init_task(
        cls,
        config: DistillationConfig,
        prompts_path: str | Path,
        storage: StorageBackend | None = None,
        force_reset: bool = False,
        confirm_reset_callback: Any = None,
        **kwargs: Any,
    ) -> DistillationPipeline:
        """Create a new isolated task workspace on GCS and return a ready pipeline."""
        from distillfw.logging_utils import TaskRunLogger

        with TaskRunLogger(command_name="init") as run_logger:
            ws = TaskWorkspace.initialize(
                config=config,
                prompts_path=prompts_path,
                storage=storage,
                teacher_callable=kwargs.get("teacher_callable"),
                force_reset=force_reset,
                confirm_reset_callback=confirm_reset_callback,
            )
            run_logger.attach_workspace(ws)
            return cls(workspace=ws, **kwargs)

    @classmethod
    def from_task_uri(
        cls,
        task_uri: str,
        storage: StorageBackend | None = None,
        **kwargs: Any,
    ) -> DistillationPipeline:
        """Attach to an existing task workspace on GCS using ONLY its URI."""
        ws = TaskWorkspace(task_uri=task_uri, storage=storage)
        return cls(workspace=ws, **kwargs)

    def get_state(self) -> TaskState:
        """Inspect current task state from `<task_uri>/task_state.json`, refreshing live Vertex AI training job status if running."""
        return self.workspace.refresh_vertex_training_status()

    def run_stage(self, stage: StageName, force_local_train: bool = False) -> dict[str, Any]:
        """Run a single specific stage against the task workspace."""
        from distillfw.logging_utils import TaskRunLogger

        with TaskRunLogger(command_name=f"run_stage_{stage.value}", workspace=self.workspace):
            self.workspace.verify_upstream_stages_completed(stage)
            if stage == StageName.DATASET_GENERATOR:
                return DatasetGenerator(
                    workspace=self.workspace, teacher_callable=self.teacher_callable
                ).run()
            if stage == StageName.DATASET_FORMATTER:
                return DatasetFormatter(
                    workspace=self.workspace, tokenizer=self.tokenizer
                ).run()
            if stage == StageName.MODEL_TRAINER:
                return ModelTrainer(
                    workspace=self.workspace, custom_train_fn=self.custom_train_fn
                ).run(force_local=force_local_train)
            if stage == StageName.MODEL_EVALUATOR:
                return ModelEvaluator(
                    workspace=self.workspace,
                    student_predict_fn=self.student_predict_fn,
                    base_student_predict_fn=self.base_student_predict_fn,
                    judge_fn=self.judge_fn,
                ).run()
            if stage == StageName.MODEL_DEPLOYER:
                return ModelDeployer(
                    workspace=self.workspace, deploy_fn=self.deploy_fn
                ).run()
            raise ValueError(f"Unknown stage: {stage}")

    def resume(self, stop_after: StageName | None = None) -> TaskState:
        """Resume pipeline execution from the first incomplete stage on GCS."""
        from distillfw.logging_utils import TaskRunLogger

        with TaskRunLogger(command_name="resume", workspace=self.workspace):
            state = self.workspace.load_state()
            for stage in ORDERED_STAGES:
                if state.stages[stage].status != StageStatus.COMPLETED:
                    self.run_stage(stage)
                    state = self.workspace.load_state()
                    if state.stages[stage].status != StageStatus.COMPLETED:
                        raise RuntimeError(
                            f"Pipeline halted: stage '{stage.value}' finished with status "
                            f"'{state.stages[stage].status.value}' instead of COMPLETED."
                        )
                if stop_after is not None and stage == stop_after:
                    break
            return state
