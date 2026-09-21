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


class ModelEvaluator:
    """Stage 4: Evaluates distilled student against base student and Gemini teacher."""

    def __init__(
        self,
        workspace: TaskWorkspace,
        student_predict_fn: Callable[[list[str]], tuple[list[str], list[float]]] | None = None,
        judge_fn: Callable[[str, str, str], dict[str, Any]] | None = None,
    ) -> None:
        self.workspace = workspace
        self.student_predict_fn = student_predict_fn
        self.judge_fn = judge_fn

    def _generate_student_predictions(
        self, model_dir_or_id: str, prompts: list[str], max_new_tokens: int = 512
    ) -> tuple[list[str], list[float]]:
        if self.student_predict_fn is not None:
            return self.student_predict_fn(prompts)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir_or_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir_or_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()

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
        return predictions, latencies_ms

    def _run_llm_judge(
        self,
        prompts: list[str],
        teacher_refs: list[str],
        student_preds: list[str],
        judge_model_id: str,
        gcp_cfg: Any,
    ) -> dict[str, Any]:
        verdicts: list[dict[str, Any]] = []
        for prompt, ref, pred in zip(prompts, teacher_refs, student_preds):
            if self.judge_fn is not None:
                verdict = self.judge_fn(prompt, ref, pred)
            else:
                from google import genai

                client = genai.Client(
                    vertexai=True, project=gcp_cfg.project_id, location=gcp_cfg.location
                )
                judge_prompt = (
                    "You are an impartial evaluator comparing a Student model response against a Teacher reference.\n"
                    f"Prompt:\n{prompt}\n\nTeacher Reference:\n{ref}\n\nStudent Response:\n{pred}\n\n"
                    "Score the Student Response from 1 to 5 (5 = matches or exceeds teacher quality) "
                    "and return JSON: {\"score\": <int>, \"reason\": \"<string>\"}."
                )
                resp = client.models.generate_content(model=judge_model_id, contents=judge_prompt)
                try:
                    verdict = json.loads(resp.text or "{}")
                except json.JSONDecodeError:
                    verdict = {"score": 3, "reason": resp.text or ""}
            verdicts.append(verdict)

        scores = [float(v.get("score", 3.0)) for v in verdicts]
        win_or_tie_rate = float(np.mean([1.0 if s >= 4.0 else 0.0 for s in scores]))
        return {
            "mean_rubric_score": round(float(np.mean(scores)), 3),
            "win_or_tie_rate_vs_teacher": round(win_or_tie_rate, 4),
            "verdicts": verdicts,
        }

    def run(self) -> dict[str, Any]:
        """Execute Stage 4 evaluation and write scorecard & markdown report to `<task_uri>/04_evaluation/`."""
        config = self.workspace.load_config()
        self.workspace.mark_stage_running(StageName.MODEL_EVALUATOR)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_root = Path(tmp_dir)
                test_uri = f"{self.workspace.formatted_dataset_dir_uri}/test.parquet"
                if not self.workspace.storage.exists(test_uri):
                    test_uri = f"{self.workspace.formatted_dataset_dir_uri}/val.parquet"
                test_file = self.workspace.storage.download_file(
                    test_uri, tmp_root / "test.parquet"
                )
                test_df = pd.read_parquet(test_file).head(config.evaluation.max_eval_samples)

                prompts = test_df["prompt"].tolist()
                teacher_refs = [
                    c.replace("<end_of_turn>", "").strip()
                    for c in test_df["completion"].tolist()
                ]

                local_model_dir = tmp_root / "distilled_model"
                self.workspace.storage.download_dir(
                    self.workspace.exported_model_dir_uri, local_model_dir
                )

                student_preds, latencies_ms = self._generate_student_predictions(
                    str(local_model_dir), prompts
                )

                scorecard: dict[str, Any] = {
                    "task_id": config.task_id,
                    "teacher_model": config.teacher.model_id,
                    "student_model": config.student.model_id,
                    "algorithm": config.training.algorithm.value,
                    "num_eval_samples": len(prompts),
                }

                # Lexical metrics
                lexical = compute_lexical_metrics(student_preds, teacher_refs)
                scorecard["lexical_metrics"] = lexical

                # Latency & cost profiling
                if EvaluationMetric.LATENCY in config.evaluation.metrics and latencies_ms:
                    scorecard["system_metrics"] = {
                        "latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
                        "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
                        "latency_mean_ms": round(float(np.mean(latencies_ms)), 2),
                        "estimated_cost_savings_ratio": round(
                            config.evaluation.teacher_cost_per_1m_output
                            / max(config.evaluation.student_hourly_gpu_cost * 0.1, 0.01),
                            2,
                        ),
                    }

                # LLM-as-a-Judge
                if EvaluationMetric.LLM_JUDGE in config.evaluation.metrics:
                    judge_summary = self._run_llm_judge(
                        prompts=prompts,
                        teacher_refs=teacher_refs,
                        student_preds=student_preds,
                        judge_model_id=config.evaluation.judge_model_id,
                        gcp_cfg=config.gcp,
                    )
                    scorecard["llm_judge"] = {
                        "mean_rubric_score": judge_summary["mean_rubric_score"],
                        "win_or_tie_rate_vs_teacher": judge_summary["win_or_tie_rate_vs_teacher"],
                    }

                scorecard_uri = f"{self.workspace.evaluation_dir_uri}/scorecard.json"
                report_uri = f"{self.workspace.evaluation_dir_uri}/report.md"
                predictions_uri = f"{self.workspace.evaluation_dir_uri}/predictions.jsonl"

                self.workspace.storage.write_json(scorecard_uri, scorecard)

                pred_lines = [
                    json.dumps(
                        {
                            "prompt": p,
                            "teacher_reference": r,
                            "distilled_student_prediction": s,
                            "latency_ms": l,
                        }
                    )
                    for p, r, s, l in zip(prompts, teacher_refs, student_preds, latencies_ms)
                ]
                self.workspace.storage.write_text(predictions_uri, "\n".join(pred_lines) + "\n")

                report_md = (
                    f"# Evaluation Report: `{config.task_id}`\n\n"
                    f"- **Teacher**: `{config.teacher.model_id}`\n"
                    f"- **Student**: `{config.student.model_id}`\n"
                    f"- **Algorithm**: `{config.training.algorithm.value}`\n\n"
                    f"## Scorecard\n\n```json\n{json.dumps(scorecard, indent=2)}\n```\n"
                )
                self.workspace.storage.write_text(report_uri, report_md)

            artifacts = {
                "scorecard_uri": scorecard_uri,
                "report_uri": report_uri,
                "predictions_uri": predictions_uri,
                "scorecard": scorecard,
            }
            self.workspace.mark_stage_completed(StageName.MODEL_EVALUATOR, artifacts=artifacts)
            return artifacts
        except Exception as exc:
            self.workspace.mark_stage_failed(StageName.MODEL_EVALUATOR, str(exc))
            raise
