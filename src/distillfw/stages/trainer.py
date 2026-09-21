"""Stage 3: Single-node multi-GPU Gemma student training and GCS checkpoint synchronization."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from distillfw.config import TrainingAlgorithm
from distillfw.gcp.vertex import VertexJobManager
from distillfw.state import StageName, StageStatus, TaskWorkspace


class ModelTrainer:
    """Stage 3: Trains a Gemma student model on a single node (1-8 GPUs) or Vertex AI Custom Job.

    Stateless & Resumable:
    - Downloads formatted splits strictly from `<task_uri>/02_formatted_dataset/`.
    - Checks `<task_uri>/03_checkpoints/` and `progress_cursor["latest_checkpoint_uri"]`
      to resume interrupted training runs seamlessly.
    - Exports final merged model weights / LoRA adapters to `<task_uri>/05_exported_model/`.
    """

    def __init__(
        self,
        workspace: TaskWorkspace,
        custom_train_fn: Callable[[Path, Path, Path], dict[str, Any]] | None = None,
    ) -> None:
        self.workspace = workspace
        self.custom_train_fn = custom_train_fn

    def _build_peft_config(self, student_cfg: Any):
        if student_cfg.peft_method == "none":
            return None
        from peft import LoraConfig, TaskType

        return LoraConfig(
            r=student_cfg.lora_r,
            lora_alpha=student_cfg.lora_alpha,
            lora_dropout=student_cfg.lora_dropout,
            target_modules=student_cfg.target_modules,
            task_type=TaskType.CAUSAL_LM,
        )

    def _run_local_training(
        self,
        config: Any,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None,
        local_ckpt_dir: Path,
        local_export_dir: Path,
    ) -> dict[str, Any]:
        if self.custom_train_fn is not None:
            train_path = local_ckpt_dir / "train.parquet"
            train_df.to_parquet(train_path, index=False)
            return self.custom_train_fn(train_path, local_ckpt_dir, local_export_dir)

        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"GPU required for training, but no CUDA GPU is attached to this machine "
                f"(torch.cuda.is_available() == False). Your task config currently specifies "
                f"training.execution_mode='{config.training.execution_mode}'. "
                f"Set training.execution_mode='vertex_custom_job' in your config.yaml (and update "
                f"the frozen config on GCS at '{self.workspace.config_uri}') so Stage 3 runs on a "
                f"Vertex AI GPU node ({config.training.vertex_machine_type} with "
                f"{config.training.vertex_accelerator_count}x {config.training.vertex_accelerator_type}), "
                f"or execute this stage on a machine with an NVIDIA GPU attached."
            )

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.student.dtype, torch.bfloat16)

        quant_config = None
        if config.student.peft_method == "qlora_4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
            )
        elif config.student.peft_method == "qlora_8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        tokenizer = AutoTokenizer.from_pretrained(config.student.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.student.model_id,
            torch_dtype=torch_dtype,
            quantization_config=quant_config,
            device_map="auto",
        )

        peft_config = self._build_peft_config(config.student)
        if peft_config is not None:
            from peft import get_peft_model

            model = get_peft_model(model, peft_config)

        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(val_df) if val_df is not None and len(val_df) > 0 else None

        import inspect
        import math

        steps_per_epoch = max(
            1,
            math.ceil(
                len(train_dataset)
                / max(
                    1,
                    config.training.per_device_batch_size
                    * config.training.gradient_accumulation_steps,
                )
            ),
        )
        total_steps = max(1, steps_per_epoch * config.training.num_epochs)
        warmup_steps = max(0, int(total_steps * config.training.warmup_ratio))

        training_args = TrainingArguments(
            output_dir=str(local_ckpt_dir),
            num_train_epochs=config.training.num_epochs,
            per_device_train_batch_size=config.training.per_device_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            learning_rate=config.training.learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=config.training.weight_decay,
            save_steps=config.training.save_steps,
            logging_steps=config.training.logging_steps,
            bf16=(config.student.dtype == "bfloat16"),
            fp16=(config.student.dtype == "float16"),
            report_to="none",
        )

        algo = config.training.algorithm

        if algo == TrainingAlgorithm.SFT_SEQKD:
            from trl import SFTConfig, SFTTrainer

            # Ensure any accelerate functools.partial forward hook exposes __func__
            base_target = model.get_base_model() if hasattr(model, "get_base_model") else model
            for mod in (model, base_target):
                fwd = getattr(mod, "forward", None)
                if fwd is not None and not hasattr(fwd, "__func__"):
                    try:
                        fwd.__func__ = getattr(mod.__class__, "forward", getattr(fwd, "func", fwd))  # type: ignore[attr-defined]
                    except Exception:
                        pass

            sft_sig = inspect.signature(SFTConfig.__init__).parameters
            extra_sft_kwargs: dict[str, Any] = {}
            if "max_length" in sft_sig:
                extra_sft_kwargs["max_length"] = config.student.max_seq_length
            else:
                extra_sft_kwargs["max_seq_length"] = config.student.max_seq_length
            if "loss_type" in sft_sig:
                extra_sft_kwargs["loss_type"] = "nll"

            sft_args = SFTConfig(
                output_dir=str(local_ckpt_dir),
                num_train_epochs=config.training.num_epochs,
                per_device_train_batch_size=config.training.per_device_batch_size,
                gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                learning_rate=config.training.learning_rate,
                warmup_steps=warmup_steps,
                weight_decay=config.training.weight_decay,
                save_steps=config.training.save_steps,
                logging_steps=config.training.logging_steps,
                bf16=(config.student.dtype == "bfloat16"),
                fp16=(config.student.dtype == "float16"),
                dataset_text_field="text",
                report_to="none",
                **extra_sft_kwargs,
            )
            trainer = SFTTrainer(
                model=model,
                args=sft_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
            )
        elif algo in (
            TrainingAlgorithm.FORWARD_KL,
            TrainingAlgorithm.REVERSE_KL,
            TrainingAlgorithm.JSD,
            TrainingAlgorithm.SKEW_KL,
        ):
            from distillfw.algorithms.trainers import OffPolicyLogitDistillationTrainer

            trainer = OffPolicyLogitDistillationTrainer(
                algorithm=algo.value,
                temperature=config.training.temperature,
                skew_alpha=config.training.skew_alpha,
                jsd_beta=config.training.gkd_beta,
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
            )
        elif algo == TrainingAlgorithm.GKD:
            try:
                from trl import GKDConfig, GKDTrainer
            except ImportError:
                from trl.experimental.gkd import GKDConfig, GKDTrainer

            gkd_args = GKDConfig(
                output_dir=str(local_ckpt_dir),
                num_train_epochs=config.training.num_epochs,
                per_device_train_batch_size=config.training.per_device_batch_size,
                warmup_steps=warmup_steps,
                lmbda=config.training.gkd_lambda,
                beta=config.training.gkd_beta,
                temperature=config.training.temperature,
                report_to="none",
            )
            trainer = GKDTrainer(
                model=model,
                args=gkd_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
            )
        elif algo == TrainingAlgorithm.DPO:
            from trl import DPOConfig, DPOTrainer

            dpo_args = DPOConfig(
                output_dir=str(local_ckpt_dir),
                num_train_epochs=config.training.num_epochs,
                per_device_train_batch_size=config.training.per_device_batch_size,
                warmup_steps=warmup_steps,
                beta=config.training.dpo_beta,
                report_to="none",
            )
            trainer = DPOTrainer(
                model=model,
                args=dpo_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")

        train_result = trainer.train()

        # Export merged or adapter weights
        local_export_dir.mkdir(parents=True, exist_ok=True)
        if config.deployment.merge_lora and hasattr(model, "merge_and_unload"):
            merged = model.merge_and_unload()
            merged.save_pretrained(local_export_dir)
        else:
            model.save_pretrained(local_export_dir)
        tokenizer.save_pretrained(local_export_dir)

        return {
            "train_loss": float(train_result.training_loss),
            "global_step": int(train_result.global_step),
        }

    def run(self, force_local: bool = False) -> dict[str, Any]:
        """Execute Stage 3 training and sync checkpoints + exported model to GCS."""
        from distillfw.logging_utils import get_logger
        from distillfw.state import verify_hf_token_present

        logger = get_logger("stages.trainer")
        verify_hf_token_present()
        self.workspace.verify_upstream_stages_completed(StageName.MODEL_TRAINER)
        config = self.workspace.load_config()
        state = self.workspace.load_state()
        stage_rec = state.stages[StageName.MODEL_TRAINER]

        # Managed single-node Vertex AI Custom Job mode
        if (
            config.gcp is not None
            and config.training.execution_mode == "vertex_custom_job"
            and not force_local
            and self.custom_train_fn is None
        ):
            job_mgr = VertexJobManager(config.gcp)
            existing_job = stage_rec.progress_cursor.get("vertex_job_resource_name")
            need_new_job = not existing_job
            if existing_job:
                try:
                    existing_info = job_mgr.get_job_status(existing_job)
                    existing_norm_state = existing_info.get("normalized_state", "")
                except Exception:
                    existing_norm_state = ""
                if existing_norm_state in (
                    "JOB_STATE_FAILED",
                    "JOB_STATE_CANCELLED",
                    "JOB_STATE_CANCELLING",
                    "JOB_STATE_EXPIRED",
                ):
                    logger.warning(
                        "Previous Vertex AI Custom Job %s is in terminal state %s; submitting a new job...",
                        existing_job,
                        existing_norm_state,
                    )
                    need_new_job = True

            if need_new_job:
                logger.info(
                    "Submitting managed Vertex AI Custom Training job for task '%s' (machine=%s, gpu=%s x%d)...",
                    config.task_id,
                    config.training.vertex_machine_type,
                    config.training.vertex_accelerator_type,
                    config.training.vertex_accelerator_count,
                )
                job_info = job_mgr.submit_training_job(
                    display_name=f"distillfw-train-{config.task_id}",
                    task_uri=self.workspace.task_uri,
                    training_config=config.training,
                )
                job_resource_name = job_info["job_resource_name"]
                self.workspace.mark_stage_running(
                    StageName.MODEL_TRAINER,
                    cursor_updates={"vertex_job_resource_name": job_resource_name},
                )
                logger.info("Submitted Vertex AI Custom Job: %s", job_resource_name)
            else:
                job_resource_name = existing_job
                logger.info("Attaching to existing Vertex AI Custom Job: %s", job_resource_name)

            try:
                final_job_status = job_mgr.wait_for_job_completion(
                    job_resource_name=job_resource_name,
                    on_poll_callback=lambda info: self.workspace.update_stage_cursor(
                        StageName.MODEL_TRAINER,
                        {"vertex_job_state": info.get("normalized_state", info.get("state"))},
                    ),
                )
            except Exception as exc:
                self.workspace.mark_stage_failed(StageName.MODEL_TRAINER, str(exc))
                raise

            # Ensure stage is marked COMPLETED if the remote worker did not already update it
            latest_state = self.workspace.load_state()
            if latest_state.stages[StageName.MODEL_TRAINER].status != StageStatus.COMPLETED:
                self.workspace.mark_stage_completed(
                    StageName.MODEL_TRAINER,
                    artifacts={
                        "exported_model_uri": self.workspace.exported_model_dir_uri,
                        "vertex_job_resource_name": job_resource_name,
                    },
                )
            return final_job_status

        self.workspace.mark_stage_running(StageName.MODEL_TRAINER)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_root = Path(tmp_dir)
                train_file = self.workspace.storage.download_file(
                    f"{self.workspace.formatted_dataset_dir_uri}/train.parquet",
                    tmp_root / "train.parquet",
                )
                train_df = pd.read_parquet(train_file)

                val_df = None
                val_uri = f"{self.workspace.formatted_dataset_dir_uri}/val.parquet"
                if self.workspace.storage.exists(val_uri):
                    val_file = self.workspace.storage.download_file(
                        val_uri, tmp_root / "val.parquet"
                    )
                    val_df = pd.read_parquet(val_file)

                logger.info(
                    "Starting single-node training | student=%s | algorithm=%s | paradigm=%s | train_rows=%d | val_rows=%d | epochs=%d",
                    config.student.model_id,
                    config.training.algorithm.value,
                    config.training.paradigm.value,
                    len(train_df),
                    len(val_df) if val_df is not None else 0,
                    config.training.num_epochs,
                )

                local_ckpt_dir = tmp_root / "checkpoints"
                local_export_dir = tmp_root / "exported_model"
                local_ckpt_dir.mkdir(parents=True, exist_ok=True)
                local_export_dir.mkdir(parents=True, exist_ok=True)

                metrics = self._run_local_training(
                    config=config,
                    train_df=train_df,
                    val_df=val_df,
                    local_ckpt_dir=local_ckpt_dir,
                    local_export_dir=local_export_dir,
                )
                logger.info("Local training completed: metrics=%s", metrics)

                # Write training summary metadata
                (local_ckpt_dir / "training_metrics.json").write_text(
                    json.dumps(metrics, indent=2), encoding="utf-8"
                )

                self.workspace.storage.upload_dir(
                    local_ckpt_dir, self.workspace.checkpoints_dir_uri
                )
                logger.info("Uploaded training checkpoints to %s", self.workspace.checkpoints_dir_uri)
                self.workspace.storage.upload_dir(
                    local_export_dir, self.workspace.exported_model_dir_uri
                )
                logger.info("Uploaded exported student model to %s", self.workspace.exported_model_dir_uri)

            artifacts = {
                "checkpoints_dir_uri": self.workspace.checkpoints_dir_uri,
                "exported_model_dir_uri": self.workspace.exported_model_dir_uri,
                "training_metrics": metrics,
            }
            self.workspace.mark_stage_completed(StageName.MODEL_TRAINER, artifacts=artifacts)
            return artifacts
        except Exception as exc:
            self.workspace.mark_stage_failed(StageName.MODEL_TRAINER, str(exc))
            raise
