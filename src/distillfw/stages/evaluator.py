"""Stage 4: Model evaluation comparing Gemini Teacher vs Base Gemma vs Distilled Gemma."""

from __future__ import annotations

import json
import math
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from distillfw.config import EvaluationMetric
from distillfw.state import StageName, TaskWorkspace


def _compute_ngram_overlap(pred: str, ref: str, n: int = 1) -> float:
    pred_tokens = pred.strip().split()
    ref_tokens = ref.strip().split()
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0.0
    pred_ngrams = Counter(tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1))
    ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
    overlap = sum((pred_ngrams & ref_ngrams).values())
    return overlap / max(sum(ref_ngrams.values()), 1)


def compute_lexical_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute Exact Match, ROUGE-1, ROUGE-2, and smoothed BLEU-4."""
    exact_matches = [
        1.0 if p.strip() == r.strip() else 0.0 for p, r in zip(predictions, references)
    ]
    rouge1 = [_compute_ngram_overlap(p, r, n=1) for p, r in zip(predictions, references)]
    rouge2 = [_compute_ngram_overlap(p, r, n=2) for p, r in zip(predictions, references)]

    # Smoothed corpus BLEU approximation
    bleu_scores = []
    for p, r in zip(predictions, references):
        p_len = max(len(p.strip().split()), 1)
        r_len = len(r.strip().split())
        bp = 1.0 if p_len > r_len else math.exp(1.0 - r_len / p_len)
        precisions = [max(_compute_ngram_overlap(p, r, n=k), 1e-4) for k in range(1, 5)]
        geo_mean = math.exp(sum(math.log(x) for x in precisions) / 4.0)
        bleu_scores.append(bp * geo_mean)

    return {
        "exact_match": round(float(np.mean(exact_matches)), 4),
        "rouge1": round(float(np.mean(rouge1)), 4),
        "rouge2": round(float(np.mean(rouge2)), 4),
        "bleu": round(float(np.mean(bleu_scores)), 4),
    }


def _compute_metrics_delta(
    before_eval: dict[str, Any], after_eval: dict[str, Any]
) -> dict[str, Any]:
    """Compute metric deltas (`after_training - before_training`) across lexical, judge, and system metrics."""
    delta: dict[str, Any] = {}

    if "lexical_metrics" in before_eval and "lexical_metrics" in after_eval:
        delta["lexical_metrics"] = {
            k: round(float(after_eval["lexical_metrics"][k]) - float(before_eval["lexical_metrics"][k]), 4)
            for k in after_eval["lexical_metrics"]
            if k in before_eval["lexical_metrics"]
        }

    if "llm_judge" in before_eval and "llm_judge" in after_eval:
        delta["llm_judge"] = {
            k: round(float(after_eval["llm_judge"][k]) - float(before_eval["llm_judge"][k]), 4)
            for k in ("mean_rubric_score", "win_or_tie_rate_vs_teacher")
            if k in after_eval["llm_judge"] and k in before_eval["llm_judge"]
        }

    if "system_metrics" in before_eval and "system_metrics" in after_eval:
        delta["system_metrics"] = {
            k: round(float(after_eval["system_metrics"][k]) - float(before_eval["system_metrics"][k]), 2)
            for k in ("latency_p50_ms", "latency_p95_ms", "latency_mean_ms")
            if k in after_eval["system_metrics"] and k in before_eval["system_metrics"]
        }

    return delta


class ModelEvaluator:
    """Stage 4: Evaluates student model before training (base) and after training (distilled) on both train and test splits."""

    def __init__(
        self,
        workspace: TaskWorkspace,
        student_predict_fn: Callable[..., tuple[list[str], list[float]]] | None = None,
        base_student_predict_fn: Callable[[list[str]], tuple[list[str], list[float]]] | None = None,
        judge_fn: Callable[[str, str, str], dict[str, Any]] | None = None,
    ) -> None:
        self.workspace = workspace
        self.student_predict_fn = student_predict_fn
        self.base_student_predict_fn = base_student_predict_fn
        self.judge_fn = judge_fn

    def _call_custom_predict_fn(
        self, prompts: list[str], model_stage: str
    ) -> tuple[list[str], list[float]]:
        import inspect

        if model_stage == "before_training" and self.base_student_predict_fn is not None:
            return self.base_student_predict_fn(prompts)

        assert self.student_predict_fn is not None
        try:
            sig = inspect.signature(self.student_predict_fn)
            params = [
                p
                for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            has_varargs = any(
                p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()
            )
            if len(params) >= 2 or has_varargs:
                return self.student_predict_fn(prompts, model_stage)
        except (ValueError, TypeError):
            pass
        return self.student_predict_fn(prompts)

    def _generate_student_predictions(
        self,
        model_dir_or_id: str,
        prompts: list[str],
        fallback_model_id: str | None = None,
        max_new_tokens: int = 512,
        model_stage: str = "after_training",
    ) -> tuple[list[str], list[float]]:
        res = self._generate_student_predictions_for_splits(
            model_dir_or_id=model_dir_or_id,
            split_prompts={"default": prompts},
            fallback_model_id=fallback_model_id,
            max_new_tokens=max_new_tokens,
            model_stage=model_stage,
        )
        return res["default"]

    def _generate_student_predictions_for_splits(
        self,
        model_dir_or_id: str,
        split_prompts: dict[str, list[str]],
        fallback_model_id: str | None = None,
        max_new_tokens: int = 512,
        model_stage: str = "after_training",
    ) -> dict[str, tuple[list[str], list[float]]]:
        """Load `model_dir_or_id` once and generate predictions across all requested splits (`train`, `test`)."""
        if self.student_predict_fn is not None or (
            model_stage == "before_training" and self.base_student_predict_fn is not None
        ):
            return {
                split_name: self._call_custom_predict_fn(prompts, model_stage=model_stage)
                for split_name, prompts in split_prompts.items()
            }

        import gc
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir_or_id)
        except Exception:
            if fallback_model_id:
                tokenizer = AutoTokenizer.from_pretrained(fallback_model_id)
            else:
                raise
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_dir_or_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()

        results: dict[str, tuple[list[str], list[float]]] = {}
        try:
            for split_name, prompts in split_prompts.items():
                predictions: list[str] = []
                latencies_ms: list[float] = []
                for prompt in prompts:
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, max_new_tokens=max_new_tokens, do_sample=False
                        )
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                    gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
                    predictions.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
                results[split_name] = (predictions, latencies_ms)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def _run_llm_judge(
        self,
        prompts: list[str],
        teacher_refs: list[str],
        student_preds: list[str],
        judge_model_id: str,
        gcp_cfg: Any,
        judge_location: str | None = None,
    ) -> dict[str, Any]:
        verdicts: list[dict[str, Any]] = []
        default_loc = gcp_cfg.location if gcp_cfg is not None else "us-central1"
        resolved_location = judge_location or default_loc
        project_id = gcp_cfg.project_id if gcp_cfg is not None else None
        client = None
        if self.judge_fn is None:
            from google import genai

            client = genai.Client(
                vertexai=True, project=project_id, location=resolved_location
            )

        for prompt, ref, pred in zip(prompts, teacher_refs, student_preds):
            if self.judge_fn is not None:
                verdict = self.judge_fn(prompt, ref, pred)
            else:
                from distillfw.gcp.retry import call_with_exponential_backoff

                assert client is not None
                judge_prompt = (
                    "You are an impartial evaluator comparing a Student model response against a Teacher reference.\n"
                    f"Prompt:\n{prompt}\n\nTeacher Reference:\n{ref}\n\nStudent Response:\n{pred}\n\n"
                    "Score the Student Response from 1 to 5 (5 = matches or exceeds teacher quality) "
                    "and return JSON: {\"score\": <int>, \"reason\": \"<string>\"}."
                )
                resp = call_with_exponential_backoff(
                    lambda: client.chats.create(model=judge_model_id).send_message(
                        message=judge_prompt
                    ),
                    max_retries=12,
                    operation_name=f"Stage 4 LLM Judge ({judge_model_id})",
                )
                try:
                    verdict = json.loads(resp.text or "{}")
                except json.JSONDecodeError:
                    verdict = {"score": 3, "reason": resp.text or ""}
            verdicts.append(verdict)

        scores = [float(v.get("score", 3.0)) for v in verdicts] if verdicts else [0.0]
        win_or_tie_rate = (
            float(np.mean([1.0 if s >= 4.0 else 0.0 for s in scores])) if verdicts else 0.0
        )
        return {
            "mean_rubric_score": round(float(np.mean(scores)), 3),
            "win_or_tie_rate_vs_teacher": round(win_or_tie_rate, 4),
            "verdicts": verdicts,
        }

    def _compute_split_stage_metrics(
        self,
        prompts: list[str],
        teacher_refs: list[str],
        student_preds: list[str],
        latencies_ms: list[float],
        config: Any,
        split_name: str,
        model_stage: str,
        logger: Any,
    ) -> dict[str, Any]:
        """Compute lexical, system latency, and LLM-as-a-Judge metrics for a single `(split_name, model_stage)` pair."""
        metrics_out: dict[str, Any] = {
            "num_samples": len(prompts),
        }

        lexical = compute_lexical_metrics(student_preds, teacher_refs)
        metrics_out["lexical_metrics"] = lexical
        logger.info(
            "[%s | %s] Computed lexical metrics: %s",
            split_name,
            model_stage,
            lexical,
        )

        if EvaluationMetric.LATENCY in config.evaluation.metrics and latencies_ms:
            metrics_out["system_metrics"] = {
                "latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
                "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
                "latency_mean_ms": round(float(np.mean(latencies_ms)), 2),
                "estimated_cost_savings_ratio": round(
                    config.evaluation.teacher_cost_per_1m_output
                    / max(config.evaluation.student_hourly_gpu_cost * 0.1, 0.01),
                    2,
                ),
            }

        if EvaluationMetric.LLM_JUDGE in config.evaluation.metrics:
            logger.info(
                "[%s | %s] Running LLM-as-a-Judge evaluation with '%s'...",
                split_name,
                model_stage,
                config.evaluation.judge_model_id,
            )
            judge_summary = self._run_llm_judge(
                prompts=prompts,
                teacher_refs=teacher_refs,
                student_preds=student_preds,
                judge_model_id=config.evaluation.judge_model_id,
                gcp_cfg=config.gcp,
                judge_location=config.evaluation.resolved_judge_location
                or config.teacher.location,
            )
            metrics_out["llm_judge"] = {
                "mean_rubric_score": judge_summary["mean_rubric_score"],
                "win_or_tie_rate_vs_teacher": judge_summary["win_or_tie_rate_vs_teacher"],
            }
            logger.info(
                "[%s | %s] LLM-as-a-Judge results: %s",
                split_name,
                model_stage,
                metrics_out["llm_judge"],
            )

        return metrics_out

    def _build_split_markdown_table(
        self,
        split_label: str,
        num_samples: int,
        before_m: dict[str, Any],
        after_m: dict[str, Any],
        delta_m: dict[str, Any],
    ) -> str:
        lines = [
            f"### {split_label} (`n={num_samples}`)",
            "",
            "| Metric | Before Training (Base Student) | After Training (Distilled Student) | Improvement (Delta) |",
            "| :--- | ---: | ---: | ---: |",
        ]
        for key, label in [
            ("exact_match", "Exact Match"),
            ("rouge1", "ROUGE-1"),
            ("rouge2", "ROUGE-2"),
            ("bleu", "BLEU-4"),
        ]:
            b_val = before_m.get("lexical_metrics", {}).get(key)
            a_val = after_m.get("lexical_metrics", {}).get(key)
            d_val = delta_m.get("lexical_metrics", {}).get(key)
            if b_val is not None and a_val is not None:
                sign = "+" if (d_val or 0.0) >= 0 else ""
                lines.append(
                    f"| **{label}** | `{b_val:.4f}` | `{a_val:.4f}` | `{sign}{d_val:.4f}` |"
                )

        if "llm_judge" in before_m and "llm_judge" in after_m:
            for key, label, fmt in [
                ("mean_rubric_score", "LLM Judge Score (1-5)", ".3f"),
                ("win_or_tie_rate_vs_teacher", "Win/Tie Rate vs Teacher", ".4f"),
            ]:
                b_val = before_m["llm_judge"].get(key)
                a_val = after_m["llm_judge"].get(key)
                d_val = delta_m.get("llm_judge", {}).get(key)
                if b_val is not None and a_val is not None:
                    sign = "+" if (d_val or 0.0) >= 0 else ""
                    lines.append(
                        f"| **{label}** | `{b_val:{fmt}}` | `{a_val:{fmt}}` | `{sign}{d_val:{fmt}}` |"
                    )

        if "system_metrics" in before_m and "system_metrics" in after_m:
            b_lat = before_m["system_metrics"].get("latency_mean_ms")
            a_lat = after_m["system_metrics"].get("latency_mean_ms")
            d_lat = delta_m.get("system_metrics", {}).get("latency_mean_ms")
            if b_lat is not None and a_lat is not None:
                sign = "+" if (d_lat or 0.0) >= 0 else ""
                lines.append(
                    f"| **Mean Latency (ms)** | `{b_lat:.2f}` | `{a_lat:.2f}` | `{sign}{d_lat:.2f}` |"
                )

        return "\n".join(lines)

    def run(self) -> dict[str, Any]:
        """Execute Stage 4 evaluation on both `train` and `test` splits before and after training."""
        from distillfw.logging_utils import get_logger
        from distillfw.state import verify_judge_model_access

        logger = get_logger("stages.evaluator")
        self.workspace.verify_upstream_stages_completed(StageName.MODEL_EVALUATOR)
        self.workspace.mark_stage_running(StageName.MODEL_EVALUATOR)

        try:
            config = self.workspace.load_config()
            verify_judge_model_access(
                config,
                storage=self.workspace.storage,
                judge_callable=self.judge_fn,
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_root = Path(tmp_dir)

                # Load both train and test splits from 02_formatted_dataset/
                train_uri = f"{self.workspace.formatted_dataset_dir_uri}/train.parquet"
                test_uri = f"{self.workspace.formatted_dataset_dir_uri}/test.parquet"
                if not self.workspace.storage.exists(test_uri):
                    val_uri = f"{self.workspace.formatted_dataset_dir_uri}/val.parquet"
                    test_uri = val_uri if self.workspace.storage.exists(val_uri) else train_uri
                if not self.workspace.storage.exists(train_uri):
                    train_uri = test_uri

                train_file = self.workspace.storage.download_file(
                    train_uri, tmp_root / "train.parquet"
                )
                test_file = self.workspace.storage.download_file(
                    test_uri, tmp_root / "test.parquet"
                )

                train_df = pd.read_parquet(train_file).head(config.evaluation.max_eval_samples)
                test_df = pd.read_parquet(test_file).head(config.evaluation.max_eval_samples)

                split_data: dict[str, dict[str, list[str]]] = {}
                for split_name, df in (("train", train_df), ("test", test_df)):
                    prompts = df["prompt"].tolist()
                    refs = [
                        c.replace("<end_of_turn>", "").strip()
                        for c in df["completion"].tolist()
                    ]
                    split_data[split_name] = {"prompts": prompts, "refs": refs}

                split_prompts = {
                    split_name: data["prompts"]
                    for split_name, data in split_data.items()
                }

                logger.info(
                    "Evaluating student BEFORE training ('%s') and AFTER training on train (%d samples) and test (%d samples) splits | metrics=%s",
                    config.student.model_id,
                    len(split_prompts["train"]),
                    len(split_prompts["test"]),
                    [m.value for m in config.evaluation.metrics],
                )

                local_model_dir = tmp_root / "distilled_model"
                if self.student_predict_fn is None:
                    exported_files = self.workspace.storage.list_uris(
                        self.workspace.exported_model_dir_uri
                    )
                    if not exported_files:
                        raise RuntimeError(
                            f"No exported model files found under '{self.workspace.exported_model_dir_uri}'. "
                            f"Ensure stage '{StageName.MODEL_TRAINER.value}' completed and exported model weights."
                        )
                self.workspace.storage.download_dir(
                    self.workspace.exported_model_dir_uri, local_model_dir
                )

                # 1. Evaluate student model BEFORE training (base student model) across train & test splits
                logger.info(
                    "Generating predictions for student model BEFORE training ('%s') on train and test splits...",
                    config.student.model_id,
                )
                before_preds_by_split = self._generate_student_predictions_for_splits(
                    model_dir_or_id=config.student.model_id,
                    split_prompts=split_prompts,
                    fallback_model_id=config.student.model_id,
                    model_stage="before_training",
                )

                # 2. Evaluate student model AFTER training (distilled model) across train & test splits
                logger.info(
                    "Generating predictions for student model AFTER training ('%s') on train and test splits...",
                    self.workspace.exported_model_dir_uri,
                )
                after_preds_by_split = self._generate_student_predictions_for_splits(
                    model_dir_or_id=str(local_model_dir),
                    split_prompts=split_prompts,
                    fallback_model_id=config.student.model_id,
                    model_stage="after_training",
                )

                before_training_metrics: dict[str, Any] = {}
                after_training_metrics: dict[str, Any] = {}
                improvement_metrics: dict[str, Any] = {}
                splits_summary: dict[str, Any] = {}
                pred_lines: list[str] = []

                for split_name in ("train", "test"):
                    prompts = split_data[split_name]["prompts"]
                    teacher_refs = split_data[split_name]["refs"]
                    base_preds, base_latencies = before_preds_by_split[split_name]
                    distilled_preds, distilled_latencies = after_preds_by_split[split_name]

                    before_m = self._compute_split_stage_metrics(
                        prompts=prompts,
                        teacher_refs=teacher_refs,
                        student_preds=base_preds,
                        latencies_ms=base_latencies,
                        config=config,
                        split_name=split_name,
                        model_stage="before_training",
                        logger=logger,
                    )
                    after_m = self._compute_split_stage_metrics(
                        prompts=prompts,
                        teacher_refs=teacher_refs,
                        student_preds=distilled_preds,
                        latencies_ms=distilled_latencies,
                        config=config,
                        split_name=split_name,
                        model_stage="after_training",
                        logger=logger,
                    )
                    delta_m = _compute_metrics_delta(before_m, after_m)

                    before_training_metrics[split_name] = before_m
                    after_training_metrics[split_name] = after_m
                    improvement_metrics[split_name] = delta_m
                    splits_summary[split_name] = {
                        "num_samples": len(prompts),
                        "before_training": before_m,
                        "after_training": after_m,
                        "improvement": delta_m,
                    }

                    for p, r, b_pred, d_pred, b_lat, d_lat in zip(
                        prompts,
                        teacher_refs,
                        base_preds,
                        distilled_preds,
                        base_latencies,
                        distilled_latencies,
                    ):
                        pred_lines.append(
                            json.dumps(
                                {
                                    "split": split_name,
                                    "prompt": p,
                                    "teacher_reference": r,
                                    "base_student_prediction": b_pred,
                                    "distilled_student_prediction": d_pred,
                                    "base_latency_ms": b_lat,
                                    "distilled_latency_ms": d_lat,
                                    "latency_ms": d_lat,
                                }
                            )
                        )

                test_after = after_training_metrics["test"]
                scorecard: dict[str, Any] = {
                    "task_id": config.task_id,
                    "teacher_model": config.teacher.model_id,
                    "student_model": config.student.model_id,
                    "algorithm": config.training.algorithm.value,
                    "num_eval_samples": len(split_prompts["test"]),
                    "num_train_eval_samples": len(split_prompts["train"]),
                    "num_test_eval_samples": len(split_prompts["test"]),
                    "before_training": before_training_metrics,
                    "after_training": after_training_metrics,
                    "improvement": improvement_metrics,
                    "splits": splits_summary,
                    "lexical_metrics": test_after["lexical_metrics"],
                }
                if "system_metrics" in test_after:
                    scorecard["system_metrics"] = test_after["system_metrics"]
                if "llm_judge" in test_after:
                    scorecard["llm_judge"] = test_after["llm_judge"]

                scorecard_uri = f"{self.workspace.evaluation_dir_uri}/scorecard.json"
                report_uri = f"{self.workspace.evaluation_dir_uri}/report.md"
                predictions_uri = f"{self.workspace.evaluation_dir_uri}/predictions.jsonl"

                self.workspace.storage.write_json(scorecard_uri, scorecard)
                self.workspace.storage.write_text(predictions_uri, "\n".join(pred_lines) + "\n")

                test_table_md = self._build_split_markdown_table(
                    split_label="Test Split (Held-Out Data)",
                    num_samples=len(split_prompts["test"]),
                    before_m=before_training_metrics["test"],
                    after_m=after_training_metrics["test"],
                    delta_m=improvement_metrics["test"],
                )
                train_table_md = self._build_split_markdown_table(
                    split_label="Train Split (Training Data)",
                    num_samples=len(split_prompts["train"]),
                    before_m=before_training_metrics["train"],
                    after_m=after_training_metrics["train"],
                    delta_m=improvement_metrics["train"],
                )

                report_md = (
                    f"# Evaluation Report: `{config.task_id}`\n\n"
                    f"- **Teacher**: `{config.teacher.model_id}`\n"
                    f"- **Student (Base -> Distilled)**: `{config.student.model_id}`\n"
                    f"- **Algorithm**: `{config.training.algorithm.value}`\n"
                    f"- **Train Eval Samples**: `{len(split_prompts['train'])}`\n"
                    f"- **Test Eval Samples**: `{len(split_prompts['test'])}`\n\n"
                    f"## Before vs. After Training Comparison\n\n"
                    f"{test_table_md}\n\n"
                    f"{train_table_md}\n\n"
                    f"## Full Scorecard\n\n```json\n{json.dumps(scorecard, indent=2)}\n```\n"
                )
                self.workspace.storage.write_text(report_uri, report_md)
                logger.info(
                    "Evaluation complete | scorecard: %s | report: %s",
                    scorecard_uri,
                    report_uri,
                )

            artifacts = {
                "scorecard_uri": scorecard_uri,
                "report_uri": report_uri,
                "predictions_uri": predictions_uri,
                "scorecard": scorecard,
            }
            self.workspace.mark_stage_completed(StageName.MODEL_EVALUATOR, artifacts=artifacts)
            return artifacts
        except BaseException as exc:
            self.workspace.mark_stage_failed(
                StageName.MODEL_EVALUATOR, str(exc) or exc.__class__.__name__
            )
            raise

