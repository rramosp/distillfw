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


def compute_sample_lexical_metrics(prediction: str, reference: str) -> dict[str, float]:
    """Compute single-sample Exact Match, ROUGE-1, ROUGE-2, and smoothed BLEU-4."""
    p_str = (prediction or "").strip()
    r_str = (reference or "").strip()
    exact_match = 1.0 if p_str == r_str else 0.0
    rouge1 = _compute_ngram_overlap(p_str, r_str, n=1)
    rouge2 = _compute_ngram_overlap(p_str, r_str, n=2)

    p_len = max(len(p_str.split()), 1)
    r_len = len(r_str.split())
    bp = 1.0 if p_len > r_len else math.exp(1.0 - r_len / p_len)
    precisions = [max(_compute_ngram_overlap(p_str, r_str, n=k), 1e-4) for k in range(1, 5)]
    geo_mean = math.exp(sum(math.log(x) for x in precisions) / 4.0)
    bleu = bp * geo_mean

    return {
        "exact_match": round(float(exact_match), 4),
        "rouge1": round(float(rouge1), 4),
        "rouge2": round(float(rouge2), 4),
        "bleu": round(float(bleu), 4),
    }


def compute_lexical_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute Exact Match, ROUGE-1, ROUGE-2, and smoothed BLEU-4."""
    if not predictions or not references:
        return {"exact_match": 0.0, "rouge1": 0.0, "rouge2": 0.0, "bleu": 0.0}
    per_sample = [compute_sample_lexical_metrics(p, r) for p, r in zip(predictions, references)]
    return {
        "exact_match": round(float(np.mean([m["exact_match"] for m in per_sample])), 4),
        "rouge1": round(float(np.mean([m["rouge1"] for m in per_sample])), 4),
        "rouge2": round(float(np.mean([m["rouge2"] for m in per_sample])), 4),
        "bleu": round(float(np.mean([m["bleu"] for m in per_sample])), 4),
    }


def count_prediction_tokens(text: str, tokenizer_obj: Any = None) -> int:
    """Count generated output tokens using a loaded Tokenizer/PreTrainedTokenizer or subword regex fallback."""
    if not text or not text.strip():
        return 0
    if tokenizer_obj is not None:
        try:
            encoded = tokenizer_obj.encode(text, add_special_tokens=False)
            if hasattr(encoded, "ids"):
                return max(1, len(encoded.ids))
            if isinstance(encoded, (list, tuple)):
                return max(1, len(encoded))
        except Exception:
            pass
    import re

    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def _load_workspace_tokenizer(workspace: TaskWorkspace) -> Any:
    """Load `<task_uri>/05_exported_model/tokenizer.json` in-memory if present in workspace storage."""
    try:
        tok_uri = f"{workspace.exported_model_dir_uri}/tokenizer.json"
        if workspace.storage.exists(tok_uri):
            from tokenizers import Tokenizer

            raw_tok_json = workspace.storage.read_text(tok_uri)
            return Tokenizer.from_str(raw_tok_json)
    except Exception:
        pass
    return None


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
            for k in (
                "mean_output_tokens",
                "time_per_output_token_ms",
                "latency_mean_ms",
                "latency_p50_ms",
                "latency_p95_ms",
            )
            if k in after_eval["system_metrics"] and k in before_eval["system_metrics"]
        }

    return delta


def _clamp_judge_score(val: Any, default: float = 3.0) -> float:
    try:
        num = float(val)
        if math.isnan(num) or math.isinf(num):
            return default
        return max(1.0, min(5.0, num))
    except (TypeError, ValueError):
        return default


def _clean_surrounding_reasoning(text: str) -> str:
    """Remove markdown code fence markers and trim whitespace from surrounding reasoning prose."""
    import re

    cleaned = re.sub(r"```(?:json|javascript|js)?", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()
    return cleaned


def parse_llm_judge_verdict(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Parse an LLM-as-a-Judge response into `{"score": float, "reason": str}`.

    Handles all response formats:
    1. Pure unfenced JSON (`{"score": 4, "reason": "..."}`)
    2. Markdown-fenced JSON (` ```json { ... } ``` ` or ` ``` { ... } ``` `)
    3. Markdown-fenced JSON embedded in the middle or at the end of a larger reasoning passage
    4. Unfenced JSON object `{ ... }` embedded in the middle of a larger reasoning passage
    5. Malformed JSON (e.g. unescaped quotes inside `"reason"`) or plain text `Score: X`
    6. Previously fallback-wrapped dicts (`{"score": 3, "reason": "```json { \"score\": 4, ... } ```"}`)
    """
    import re

    if isinstance(raw, dict):
        reason_field = str(raw.get("reason", "") or "").strip()
        # Check if `reason` itself contains an unparsed JSON or fenced verdict from an earlier fallback
        if reason_field and (
            "```" in reason_field
            or re.search(r'"score"\s*:\s*[1-5]', reason_field, flags=re.IGNORECASE)
        ):
            reparsed = parse_llm_judge_verdict(reason_field)
            if reparsed.get("_parsed_ok"):
                return {"score": reparsed["score"], "reason": reparsed["reason"]}
        return {
            "score": _clamp_judge_score(raw.get("score", 3.0)),
            "reason": _clean_surrounding_reasoning(reason_field),
        }

    text = str(raw or "").strip()
    if not text:
        return {"score": 3.0, "reason": ""}

    # Strategy A: Direct JSON parse
    try:
        direct_obj = json.loads(text)
        if isinstance(direct_obj, dict) and "score" in direct_obj:
            return {
                "score": _clamp_judge_score(direct_obj["score"]),
                "reason": str(direct_obj.get("reason", "") or "").strip(),
                "_parsed_ok": True,
            }
    except Exception:
        pass

    decoder = json.JSONDecoder()

    # Strategy B: Markdown fenced blocks anywhere in the text (start, middle, or end)
    fence_pattern = re.compile(r"```(?:json|javascript|js)?\s*([\s\S]*?)```", re.IGNORECASE)
    fence_matches = list(fence_pattern.finditer(text))
    for match in reversed(fence_matches):
        block_content = match.group(1).strip()
        outside_text = _clean_surrounding_reasoning(
            (text[: match.start()] + "\n" + text[match.end() :]).strip()
        )
        # Try direct parse of block content or raw_decode inside block content
        parsed_block: dict[str, Any] | None = None
        try:
            candidate = json.loads(block_content)
            if isinstance(candidate, dict) and "score" in candidate:
                parsed_block = candidate
        except Exception:
            for idx, ch in enumerate(block_content):
                if ch == "{":
                    try:
                        candidate, _ = decoder.raw_decode(block_content, idx)
                        if isinstance(candidate, dict) and "score" in candidate:
                            parsed_block = candidate
                            break
                    except Exception:
                        continue

        if parsed_block is not None:
            score = _clamp_judge_score(parsed_block["score"])
            inner_reason = str(parsed_block.get("reason", "") or "").strip()
            reason = inner_reason if inner_reason else outside_text
            return {"score": score, "reason": reason, "_parsed_ok": True}

    # Strategy C: Balanced JSON object scanner (`raw_decode`) across every `{` in text
    best_candidate: tuple[dict[str, Any], int, int] | None = None
    for idx, ch in enumerate(text):
        if ch == "{":
            try:
                candidate, end_idx = decoder.raw_decode(text, idx)
                if isinstance(candidate, dict) and "score" in candidate:
                    best_candidate = (candidate, idx, end_idx)
            except Exception:
                continue

    if best_candidate is not None:
        obj, start_idx, end_idx = best_candidate
        score = _clamp_judge_score(obj["score"])
        inner_reason = str(obj.get("reason", "") or "").strip()
        outside_text = _clean_surrounding_reasoning(
            (text[:start_idx] + "\n" + text[end_idx:]).strip()
        )
        reason = inner_reason if inner_reason else outside_text
        return {"score": score, "reason": reason, "_parsed_ok": True}

    # Strategy D: Regex fallback (for malformed JSON with unescaped quotes or plain-text verdicts)
    score_match = re.search(
        r'"score"\s*:\s*([1-5](?:\.\d+)?)|\bscore\s*[:=]\s*([1-5](?:\.\d+)?)(?:\s*/\s*5)?',
        text,
        flags=re.IGNORECASE,
    )
    if score_match:
        raw_score = score_match.group(1) or score_match.group(2)
        score = _clamp_judge_score(raw_score)
        reason_match = re.search(
            r'"reason"\s*:\s*"([\s\S]*?)"\s*(?:,\s*"\w+"|\}\s*(?:```|$))',
            text,
            flags=re.IGNORECASE,
        )
        if reason_match:
            extracted_reason = reason_match.group(1).replace('\\"', '"').strip()
            outside = _clean_surrounding_reasoning(fence_pattern.sub("", text))
            if outside and extracted_reason not in outside and not outside.startswith("{"):
                reason = f"{extracted_reason}\n\n{outside}".strip()
            else:
                reason = extracted_reason
        else:
            cleaned = _clean_surrounding_reasoning(text)
            cleaned = re.sub(r'\{\s*"score"\s*:\s*[1-5](?:\.\d+)?\s*,?\s*"reason"\s*:\s*"?', "", cleaned)
            cleaned = re.sub(r'"?\s*\}\s*$', "", cleaned).strip()
            reason = cleaned
        return {"score": score, "reason": reason, "_parsed_ok": True}

    return {"score": 3.0, "reason": _clean_surrounding_reasoning(text)}


def repair_evaluation_judge_artifacts(workspace: TaskWorkspace) -> bool:
    """Inspect `<task_uri>/04_evaluation/predictions.jsonl` and `scorecard.json`,
    re-parse any fenced/embedded JSON LLM-judge verdicts, recompute split-level
    `llm_judge` metrics (`mean_rubric_score`, `win_or_tie_rate_vs_teacher`) and deltas,
    and persist repaired artifacts back to storage if changes were detected.
    """
    predictions_uri = f"{workspace.evaluation_dir_uri}/predictions.jsonl"
    scorecard_uri = f"{workspace.evaluation_dir_uri}/scorecard.json"
    if not workspace.storage.exists(predictions_uri):
        return False

    raw_lines = [
        line for line in workspace.storage.read_text(predictions_uri).splitlines() if line.strip()
    ]
    if not raw_lines:
        return False

    records = [json.loads(line) for line in raw_lines]
    predictions_changed = False
    workspace_tokenizer = _load_workspace_tokenizer(workspace)

    split_scores: dict[str, dict[str, list[float]]] = {
        "train": {"before_training": [], "after_training": []},
        "test": {"before_training": [], "after_training": []},
    }
    split_tokens: dict[str, dict[str, list[int]]] = {
        "train": {"before_training": [], "after_training": []},
        "test": {"before_training": [], "after_training": []},
    }
    split_tpots: dict[str, dict[str, list[float]]] = {
        "train": {"before_training": [], "after_training": []},
        "test": {"before_training": [], "after_training": []},
    }

    for rec in records:
        split_name = str(rec.get("split", "test")).lower()
        if split_name not in split_scores:
            split_scores[split_name] = {"before_training": [], "after_training": []}
            split_tokens[split_name] = {"before_training": [], "after_training": []}
            split_tpots[split_name] = {"before_training": [], "after_training": []}

        for metrics_key, stage_key, pred_key, lat_key in (
            ("base_metrics", "before_training", "base_student_prediction", "base_latency_ms"),
            ("distilled_metrics", "after_training", "distilled_student_prediction", "distilled_latency_ms"),
        ):
            m_dict = rec.get(metrics_key)
            if not isinstance(m_dict, dict):
                m_dict = compute_sample_lexical_metrics(
                    str(rec.get(pred_key, "")), str(rec.get("teacher_reference", ""))
                )
                rec[metrics_key] = m_dict
                predictions_changed = True

            pred_text = str(rec.get(pred_key, ""))
            lat_val = m_dict.get("latency_ms", rec.get(lat_key))
            tok_count = m_dict.get("output_tokens")
            if tok_count is None or int(tok_count) <= 0:
                tok_count = count_prediction_tokens(pred_text, workspace_tokenizer)
                if m_dict.get("output_tokens") != tok_count:
                    m_dict["output_tokens"] = tok_count
                    predictions_changed = True
            else:
                tok_count = int(tok_count)

            split_tokens[split_name][stage_key].append(tok_count)

            if lat_val is not None:
                try:
                    lat_float = round(float(lat_val), 2)
                    if m_dict.get("latency_ms") != lat_float:
                        m_dict["latency_ms"] = lat_float
                        predictions_changed = True
                    tpot_val = round(float(lat_val) / max(tok_count, 1), 2)
                    if m_dict.get("ms_per_output_token") != tpot_val:
                        m_dict["ms_per_output_token"] = tpot_val
                        predictions_changed = True
                    split_tpots[split_name][stage_key].append(float(lat_val) / max(tok_count, 1))
                except (TypeError, ValueError):
                    pass

            if "llm_judge_reason" in m_dict or "llm_judge_score" in m_dict:
                old_score = m_dict.get("llm_judge_score")
                old_reason = m_dict.get("llm_judge_reason", "")
                parsed = parse_llm_judge_verdict(
                    {"score": old_score if old_score is not None else 3.0, "reason": old_reason}
                )
                new_score = parsed["score"]
                new_reason = parsed["reason"]
                if old_score != new_score or old_reason != new_reason:
                    m_dict["llm_judge_score"] = new_score
                    m_dict["llm_judge_reason"] = new_reason
                    predictions_changed = True
                split_scores[split_name][stage_key].append(float(new_score))

    if predictions_changed:
        workspace.storage.write_text(
            predictions_uri,
            "\n".join(json.dumps(r) for r in records) + "\n",
        )

    if not workspace.storage.exists(scorecard_uri):
        return predictions_changed

    scorecard = workspace.storage.read_json(scorecard_uri)
    scorecard_changed = False

    for split_name in ("train", "test"):
        b_scores = split_scores.get(split_name, {}).get("before_training", [])
        a_scores = split_scores.get(split_name, {}).get("after_training", [])
        b_toks = split_tokens.get(split_name, {}).get("before_training", [])
        a_toks = split_tokens.get(split_name, {}).get("after_training", [])
        b_tpots = split_tpots.get(split_name, {}).get("before_training", [])
        a_tpots = split_tpots.get(split_name, {}).get("after_training", [])
        if not b_scores and not a_scores and not b_toks and not a_toks:
            continue

        def _calc_judge_summary(scores_list: list[float]) -> dict[str, float]:
            mean_s = round(float(np.mean(scores_list)), 3)
            win_tie = round(float(np.mean([1.0 if s >= 4.0 else 0.0 for s in scores_list])), 4)
            return {"mean_rubric_score": mean_s, "win_or_tie_rate_vs_teacher": win_tie}

        b_summary = _calc_judge_summary(b_scores) if b_scores else None
        a_summary = _calc_judge_summary(a_scores) if a_scores else None

        before_split = scorecard.get("before_training", {}).get(split_name)
        if not isinstance(before_split, dict):
            before_split = scorecard.get("splits", {}).get(split_name, {}).get("before_training")
        after_split = scorecard.get("after_training", {}).get(split_name)
        if not isinstance(after_split, dict):
            after_split = scorecard.get("splits", {}).get(split_name, {}).get("after_training")

        if isinstance(before_split, dict) and isinstance(before_split.get("system_metrics"), dict):
            if b_toks:
                mean_b_toks = round(float(np.mean(b_toks)), 2)
                if before_split["system_metrics"].get("mean_output_tokens") != mean_b_toks:
                    before_split["system_metrics"]["mean_output_tokens"] = mean_b_toks
                    scorecard_changed = True
            if b_tpots:
                mean_b_tpot = round(float(np.mean(b_tpots)), 2)
                if before_split["system_metrics"].get("time_per_output_token_ms") != mean_b_tpot:
                    before_split["system_metrics"]["time_per_output_token_ms"] = mean_b_tpot
                    scorecard_changed = True

        if isinstance(after_split, dict) and isinstance(after_split.get("system_metrics"), dict):
            if a_toks:
                mean_a_toks = round(float(np.mean(a_toks)), 2)
                if after_split["system_metrics"].get("mean_output_tokens") != mean_a_toks:
                    after_split["system_metrics"]["mean_output_tokens"] = mean_a_toks
                    scorecard_changed = True
            if a_tpots:
                mean_a_tpot = round(float(np.mean(a_tpots)), 2)
                if after_split["system_metrics"].get("time_per_output_token_ms") != mean_a_tpot:
                    after_split["system_metrics"]["time_per_output_token_ms"] = mean_a_tpot
                    scorecard_changed = True

        if isinstance(before_split, dict) and b_summary is not None:
            if before_split.get("llm_judge") != b_summary:
                before_split["llm_judge"] = b_summary
                scorecard_changed = True
        if isinstance(after_split, dict) and a_summary is not None:
            if after_split.get("llm_judge") != a_summary:
                after_split["llm_judge"] = a_summary
                scorecard_changed = True

        if isinstance(before_split, dict) and isinstance(after_split, dict):
            new_delta = _compute_metrics_delta(before_split, after_split)
            if isinstance(scorecard.get("improvement"), dict) and split_name in scorecard.get("improvement", {}):
                if scorecard["improvement"].get(split_name) != new_delta:
                    scorecard["improvement"][split_name] = new_delta
                    scorecard_changed = True
            elif "before_training" in scorecard or "splits" in scorecard:
                if scorecard.setdefault("improvement", {}).get(split_name) != new_delta:
                    scorecard["improvement"][split_name] = new_delta
                    scorecard_changed = True

            if split_name in scorecard.get("splits", {}):
                scorecard["splits"][split_name]["before_training"] = before_split
                scorecard["splits"][split_name]["after_training"] = after_split
                scorecard["splits"][split_name]["improvement"] = new_delta

        if split_name == "test":
            if (
                isinstance(after_split, dict)
                and isinstance(after_split.get("system_metrics"), dict)
                and "system_metrics" in scorecard
                and scorecard.get("system_metrics") != after_split["system_metrics"]
            ):
                scorecard["system_metrics"] = dict(after_split["system_metrics"])
                scorecard_changed = True
            if a_summary is not None and scorecard.get("llm_judge") != a_summary and "llm_judge" in scorecard:
                scorecard["llm_judge"] = a_summary
                scorecard_changed = True
            if isinstance(scorecard.get("base_student"), dict) and b_summary is not None:
                if scorecard["base_student"].get("llm_judge") != b_summary:
                    scorecard["base_student"]["llm_judge"] = b_summary
                    scorecard_changed = True
            if isinstance(scorecard.get("distilled_student"), dict) and a_summary is not None:
                if scorecard["distilled_student"].get("llm_judge") != a_summary:
                    scorecard["distilled_student"]["llm_judge"] = a_summary
                    scorecard_changed = True
            if (
                isinstance(scorecard.get("improvement"), dict)
                and b_summary is not None
                and a_summary is not None
                and (
                    "llm_judge_score_delta" in scorecard["improvement"]
                    or "base_student" in scorecard
                )
            ):
                score_delta = round(
                    a_summary["mean_rubric_score"] - b_summary["mean_rubric_score"], 3
                )
                win_delta = round(
                    a_summary["win_or_tie_rate_vs_teacher"]
                    - b_summary["win_or_tie_rate_vs_teacher"],
                    4,
                )
                if (
                    scorecard["improvement"].get("llm_judge_score_delta") != score_delta
                    or scorecard["improvement"].get("llm_judge_win_rate_delta") != win_delta
                ):
                    scorecard["improvement"]["llm_judge_score_delta"] = score_delta
                    scorecard["improvement"]["llm_judge_win_rate_delta"] = win_delta
                    scorecard_changed = True

    if scorecard_changed:
        workspace.storage.write_json(scorecard_uri, scorecard)
        try:
            state = workspace.load_state()
            eval_rec = state.stages.get(StageName.MODEL_EVALUATOR)
            if eval_rec is not None and isinstance(eval_rec.artifacts, dict):
                eval_rec.artifacts["scorecard"] = scorecard
                workspace.save_state(state)
        except Exception:
            pass

    return predictions_changed or scorecard_changed



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
        judge_gen_cfg = None
        if self.judge_fn is None:
            from google import genai
            from google.genai import types

            client = genai.Client(
                vertexai=True, project=project_id, location=resolved_location
            )
            judge_gen_cfg = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=512,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )

        for prompt, ref, pred in zip(prompts, teacher_refs, student_preds):
            if self.judge_fn is not None:
                raw_verdict = self.judge_fn(prompt, ref, pred)
                if isinstance(raw_verdict, str):
                    parsed = parse_llm_judge_verdict(raw_verdict)
                    verdict = {"score": parsed["score"], "reason": parsed["reason"]}
                elif isinstance(raw_verdict, dict):
                    raw_reason = str(raw_verdict.get("reason", ""))
                    if "score" not in raw_verdict or (
                        float(raw_verdict.get("score", 3.0)) == 3.0
                        and ("```" in raw_reason or '"score"' in raw_reason)
                    ):
                        parsed = parse_llm_judge_verdict(raw_reason)
                        if parsed.get("_parsed_ok"):
                            verdict = {"score": parsed["score"], "reason": parsed["reason"]}
                        else:
                            verdict = {
                                "score": _clamp_judge_score(raw_verdict.get("score", 3.0)),
                                "reason": raw_reason,
                            }
                    else:
                        verdict = {
                            "score": _clamp_judge_score(raw_verdict.get("score", 3.0)),
                            "reason": raw_reason,
                        }
                else:
                    verdict = {"score": 3.0, "reason": ""}
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
                    lambda: client.chats.create(
                        model=judge_model_id,
                        config=judge_gen_cfg,
                    ).send_message(
                        message=judge_prompt
                    ),
                    max_retries=12,
                    operation_name=f"Stage 4 LLM Judge ({judge_model_id})",
                )
                parsed = parse_llm_judge_verdict(resp.text or "")
                verdict = {"score": parsed["score"], "reason": parsed["reason"]}
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
            workspace_tokenizer = _load_workspace_tokenizer(self.workspace)
            token_counts = [
                count_prediction_tokens(pred, workspace_tokenizer) for pred in student_preds
            ]
            tpot_ms_list = [
                float(lat) / max(int(toks), 1)
                for lat, toks in zip(latencies_ms, token_counts)
            ]
            metrics_out["system_metrics"] = {
                "mean_output_tokens": round(float(np.mean(token_counts)), 2),
                "time_per_output_token_ms": round(float(np.mean(tpot_ms_list)), 2),
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
            metrics_out["_judge_verdicts"] = judge_summary.get("verdicts", [])
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
            for key, label in [
                ("mean_output_tokens", "Output Length (Tokens)"),
                ("time_per_output_token_ms", "Time per Output Token (ms/tok)"),
                ("latency_mean_ms", "Mean Latency (ms)"),
                ("latency_p50_ms", "p50 Latency (ms)"),
                ("latency_p95_ms", "p95 Latency (ms)"),
            ]:
                b_val = before_m["system_metrics"].get(key)
                a_val = after_m["system_metrics"].get(key)
                d_val = delta_m.get("system_metrics", {}).get(key)
                if b_val is not None and a_val is not None:
                    sign = "+" if (d_val or 0.0) >= 0 else ""
                    lines.append(
                        f"| **{label}** | `{b_val:.2f}` | `{a_val:.2f}` | `{sign}{d_val:.2f}` |"
                    )

        return "\n".join(lines)

    @staticmethod
    def _emit_local_evaluation_warning(logger: Any) -> list[str]:
        """Emit a 5-line warning via both CLI logger and console printout when running evaluation locally."""
        warning_lines = [
            "WARNING [1/5]: Evaluation hardware options (vertex_machine_type, vertex_accelerator_type, vertex_accelerator_count) are not set in config.yaml.",
            "WARNING [2/5]: Stage 4 (model_evaluator) will execute locally on the current machine instead of a managed Vertex AI GPU Custom Job.",
            "WARNING [3/5]: Running student model inference (before and after training) on local CPU/hardware may be significantly slower and distort latency metrics.",
            "WARNING [4/5]: To run evaluation on dedicated GCP hardware (e.g., 1x NVIDIA_L4), configure vertex_machine_type, vertex_accelerator_type, and vertex_accelerator_count under 'evaluation:'.",
            "WARNING [5/5]: Proceeding automatically with local evaluation execution (no confirmation required)...",
        ]
        for line in warning_lines:
            logger.warning(line)
            print(line)
        return warning_lines

    def run(self, force_local: bool = False) -> dict[str, Any]:
        """Execute Stage 4 evaluation on both `train` and `test` splits before and after training."""
        from distillfw.gcp.vertex import VertexJobManager
        from distillfw.logging_utils import get_logger
        from distillfw.state import verify_hf_token_present, verify_judge_model_access

        logger = get_logger("stages.evaluator")
        self.workspace.verify_upstream_stages_completed(StageName.MODEL_EVALUATOR)

        try:
            config = self.workspace.load_config()
            state = self.workspace.load_state()
            stage_rec = state.stages[StageName.MODEL_EVALUATOR]

            # Managed single-node Vertex AI Custom Job mode when vertex_machine_type / accelerator options are set
            if (
                config.gcp is not None
                and config.evaluation.has_vertex_custom_job_config
                and not force_local
                and self.student_predict_fn is None
            ):
                verify_hf_token_present()
                verify_judge_model_access(
                    config,
                    storage=self.workspace.storage,
                    judge_callable=self.judge_fn,
                )
                job_mgr = VertexJobManager(config.gcp)
                existing_job = stage_rec.progress_cursor.get("vertex_job_resource_name")
                need_new_job = not existing_job
                if existing_job:
                    try:
                        existing_info = job_mgr.get_job_status(existing_job)
                        existing_norm_state = existing_info.get("normalized_state", "") or existing_info.get("state", "")
                    except Exception:
                        existing_norm_state = ""
                    if existing_norm_state in (
                        "JOB_STATE_FAILED",
                        "JOB_STATE_CANCELLED",
                        "JOB_STATE_CANCELLING",
                        "JOB_STATE_EXPIRED",
                    ):
                        logger.warning(
                            "Previous Vertex AI Evaluation Custom Job %s is in terminal state %s; submitting a new job...",
                            existing_job,
                            existing_norm_state,
                        )
                        need_new_job = True

                if need_new_job:
                    logger.info(
                        "Submitting managed Vertex AI Custom Evaluation job for task '%s' (machine=%s, gpu=%s x%d)...",
                        config.task_id,
                        config.evaluation.vertex_machine_type,
                        config.evaluation.vertex_accelerator_type,
                        config.evaluation.vertex_accelerator_count,
                    )
                    job_info = job_mgr.submit_evaluation_job(
                        display_name=f"distillfw-eval-{config.task_id}",
                        task_uri=self.workspace.task_uri,
                        evaluation_config=config.evaluation,
                    )
                    job_resource_name = job_info["job_resource_name"]
                    self.workspace.mark_stage_running(
                        StageName.MODEL_EVALUATOR,
                        cursor_updates={"vertex_job_resource_name": job_resource_name},
                    )
                else:
                    job_resource_name = existing_job
                    logger.info(
                        "Attached to existing Vertex AI Custom Evaluation job: %s",
                        job_resource_name,
                    )
                    self.workspace.mark_stage_running(
                        StageName.MODEL_EVALUATOR,
                        cursor_updates={"vertex_job_resource_name": job_resource_name},
                    )

                def _on_poll(status_info: dict[str, Any]) -> None:
                    self.workspace.mark_stage_running(
                        StageName.MODEL_EVALUATOR,
                        cursor_updates={
                            "vertex_job_resource_name": job_resource_name,
                            "vertex_job_state": status_info.get("normalized_state", status_info.get("state")),
                        },
                    )

                final_status = job_mgr.wait_for_job_completion(
                    job_resource_name=job_resource_name,
                    on_poll_callback=_on_poll,
                )

                scorecard_uri = f"{self.workspace.evaluation_dir_uri}/scorecard.json"
                report_uri = f"{self.workspace.evaluation_dir_uri}/report.md"
                predictions_uri = f"{self.workspace.evaluation_dir_uri}/predictions.jsonl"
                scorecard = (
                    self.workspace.storage.read_json(scorecard_uri)
                    if self.workspace.storage.exists(scorecard_uri)
                    else {}
                )

                artifacts = {
                    "scorecard_uri": scorecard_uri,
                    "report_uri": report_uri,
                    "predictions_uri": predictions_uri,
                    "scorecard": scorecard,
                    "vertex_job_resource_name": job_resource_name,
                    "vertex_job_state": final_status.get("normalized_state", final_status.get("state")),
                }
                self.workspace.mark_stage_completed(StageName.MODEL_EVALUATOR, artifacts=artifacts)
                return artifacts

            if not config.evaluation.has_vertex_custom_job_config and not force_local:
                self._emit_local_evaluation_warning(logger)

            self.workspace.mark_stage_running(StageName.MODEL_EVALUATOR)
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
                    max_new_tokens=config.teacher.max_output_tokens,
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
                    max_new_tokens=config.teacher.max_output_tokens,
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
                    base_verdicts = before_m.pop("_judge_verdicts", [])
                    distilled_verdicts = after_m.pop("_judge_verdicts", [])
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

                    workspace_tokenizer = _load_workspace_tokenizer(self.workspace)
                    for sample_idx, (p, r, b_pred, d_pred, b_lat, d_lat) in enumerate(
                        zip(
                            prompts,
                            teacher_refs,
                            base_preds,
                            distilled_preds,
                            base_latencies,
                            distilled_latencies,
                        )
                    ):
                        b_sample_metrics: dict[str, Any] = compute_sample_lexical_metrics(b_pred, r)
                        d_sample_metrics: dict[str, Any] = compute_sample_lexical_metrics(d_pred, r)
                        b_tok_count = count_prediction_tokens(b_pred, workspace_tokenizer)
                        d_tok_count = count_prediction_tokens(d_pred, workspace_tokenizer)
                        b_sample_metrics["output_tokens"] = b_tok_count
                        d_sample_metrics["output_tokens"] = d_tok_count
                        if b_lat is not None:
                            b_sample_metrics["latency_ms"] = round(float(b_lat), 2)
                            b_sample_metrics["ms_per_output_token"] = round(
                                float(b_lat) / max(b_tok_count, 1), 2
                            )
                        if d_lat is not None:
                            d_sample_metrics["latency_ms"] = round(float(d_lat), 2)
                            d_sample_metrics["ms_per_output_token"] = round(
                                float(d_lat) / max(d_tok_count, 1), 2
                            )
                        if sample_idx < len(base_verdicts) and isinstance(base_verdicts[sample_idx], dict):
                            b_verdict = base_verdicts[sample_idx]
                            if "score" in b_verdict:
                                b_sample_metrics["llm_judge_score"] = b_verdict["score"]
                            if "reason" in b_verdict:
                                b_sample_metrics["llm_judge_reason"] = b_verdict["reason"]
                        if sample_idx < len(distilled_verdicts) and isinstance(distilled_verdicts[sample_idx], dict):
                            d_verdict = distilled_verdicts[sample_idx]
                            if "score" in d_verdict:
                                d_sample_metrics["llm_judge_score"] = d_verdict["score"]
                            if "reason" in d_verdict:
                                d_sample_metrics["llm_judge_reason"] = d_verdict["reason"]

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
                                    "base_metrics": b_sample_metrics,
                                    "distilled_metrics": d_sample_metrics,
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

