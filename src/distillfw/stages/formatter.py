"""Stage 2: Dataset formatting, chat templating, loss masking, and split generation."""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from distillfw.config import DatasetRepresentation, PromptFormat
from distillfw.state import StageName, TaskWorkspace


def render_gemma_prompt(
    prompt: str,
    completion: str,
    prompt_format: PromptFormat,
    thought: str | None = None,
    system_instruction: str | None = None,
) -> tuple[str, str]:
    """Render `(formatted_prompt_prefix, formatted_target_completion)` for Gemma models."""
    if prompt_format == PromptFormat.PRETRAIN:
        return prompt, completion

    user_content = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
    prefix = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"

    if prompt_format == PromptFormat.REASONING and thought:
        target = f"<start_of_turn>thought\n{thought}<end_of_turn>\n{completion}<end_of_turn>"
    else:
        target = f"{completion}<end_of_turn>"

    return prefix, target


class DatasetFormatter:
    """Stage 2: Reads raw teacher shards from GCS and writes formatted train/val/test splits."""

    def __init__(self, workspace: TaskWorkspace, tokenizer: Any | None = None) -> None:
        self.workspace = workspace
        self.tokenizer = tokenizer

    def _get_tokenizer(self, student_model_id: str) -> Any:
        if self.tokenizer is not None:
            return self.tokenizer
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(student_model_id)

    def _format_single_record(
        self,
        raw: dict[str, Any],
        prompt_format: PromptFormat,
        representation: DatasetRepresentation,
        max_seq_length: int,
        system_instruction: str | None,
        tokenizer: Any | None,
    ) -> dict[str, Any]:
        prompt = raw["prompt"]
        completion = raw["completion"]
        thought = raw.get("thought")
        candidates = raw.get("candidates") or [completion]

        prefix, target = render_gemma_prompt(
            prompt=prompt,
            completion=completion,
            prompt_format=prompt_format,
            thought=thought,
            system_instruction=system_instruction,
        )
        full_text = prefix + target

        out: dict[str, Any] = {
            "prompt": prefix,
            "completion": target,
            "text": full_text,
        }

        if representation == DatasetRepresentation.PREFERENCE:
            rejected = candidates[-1] if len(candidates) > 1 else ""
            _, rejected_target = render_gemma_prompt(
                prompt=prompt,
                completion=rejected,
                prompt_format=prompt_format,
                system_instruction=system_instruction,
            )
            out["chosen"] = target
            out["rejected"] = rejected_target

        elif representation in (DatasetRepresentation.TOKENS, DatasetRepresentation.SPARSE_LOGPROBS):
            if tokenizer is not None:
                prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
                target_ids = tokenizer.encode(target, add_special_tokens=False)
                input_ids = (prefix_ids + target_ids)[:max_seq_length]
                labels = ([-100] * len(prefix_ids) + target_ids)[:max_seq_length]
                attention_mask = [1] * len(input_ids)
                out["input_ids"] = input_ids
                out["attention_mask"] = attention_mask
                out["labels"] = labels
            if representation == DatasetRepresentation.SPARSE_LOGPROBS:
                out["topk_logprobs"] = raw.get("topk_logprobs") or []

        return out

    def run(self) -> dict[str, Any]:
        """Execute Stage 2 formatting and persist train/val/test splits to `<task_uri>/02_formatted_dataset/`."""
        from distillfw.logging_utils import get_logger

        logger = get_logger("stages.formatter")
        self.workspace.verify_upstream_stages_completed(StageName.DATASET_FORMATTER)
        config = self.workspace.load_config()
        self.workspace.mark_stage_running(StageName.DATASET_FORMATTER)

        try:
            shard_uris = self.workspace.storage.list_uris(self.workspace.raw_dataset_dir_uri)
            raw_records: list[dict[str, Any]] = []
            for uri in shard_uris:
                if uri.endswith(".jsonl"):
                    text = self.workspace.storage.read_text(uri)
                    raw_records.extend(
                        json.loads(line) for line in text.splitlines() if line.strip()
                    )

            if not raw_records:
                raise RuntimeError(
                    f"No raw teacher records found in {self.workspace.raw_dataset_dir_uri}"
                )

            logger.info(
                "Loaded %d raw teacher record(s) from %d shard file(s) | prompt_format=%s | representation=%s",
                len(raw_records),
                len([u for u in shard_uris if u.endswith(".jsonl")]),
                config.formatting.prompt_format.value,
                config.formatting.dataset_representation.value,
            )

            tokenizer = None
            if config.formatting.dataset_representation in (
                DatasetRepresentation.TOKENS,
                DatasetRepresentation.SPARSE_LOGPROBS,
            ):
                logger.info("Loading tokenizer for student model '%s'...", config.student.model_id)
                tokenizer = self._get_tokenizer(config.student.model_id)

            formatted = [
                self._format_single_record(
                    raw=r,
                    prompt_format=config.formatting.prompt_format,
                    representation=config.formatting.dataset_representation,
                    max_seq_length=config.student.max_seq_length,
                    system_instruction=config.teacher.system_instruction,
                    tokenizer=tokenizer,
                )
                for r in raw_records
            ]

            rng = random.Random(config.formatting.seed)
            rng.shuffle(formatted)

            n = len(formatted)
            if n >= 2:
                n_test = max(1, min(n - 1, int(round(n * config.formatting.test_split_ratio))))
                remaining = n - n_test
                if config.formatting.val_split_ratio > 0 and remaining >= 2:
                    n_val = min(
                        remaining - 1,
                        int(round(n * config.formatting.val_split_ratio)),
                    )
                else:
                    n_val = 0
                n_train = remaining - n_val
                splits = {
                    "train": formatted[:n_train],
                    "val": formatted[n_train : n_train + n_val],
                    "test": formatted[n_train + n_val :],
                }
            else:
                splits = {
                    "train": formatted[:1],
                    "val": [],
                    "test": formatted[:1],
                }

            split_uris: dict[str, str] = {}
            with tempfile.TemporaryDirectory() as tmp_dir:
                for split_name, rows in splits.items():
                    if not rows:
                        continue
                    rows_with_split = [{**row, "split": split_name} for row in rows]
                    local_parquet = Path(tmp_dir) / f"{split_name}.parquet"
                    pd.DataFrame(rows_with_split).to_parquet(local_parquet, index=False)
                    dest_uri = f"{self.workspace.formatted_dataset_dir_uri}/{split_name}.parquet"
                    self.workspace.storage.upload_file(local_parquet, dest_uri)
                    split_uris[split_name] = dest_uri
                    logger.info("Uploaded split '%s' (%d rows) -> %s", split_name, len(rows), dest_uri)

            artifacts = {
                "formatted_dataset_dir_uri": self.workspace.formatted_dataset_dir_uri,
                "split_uris": split_uris,
                "split_counts": {k: len(v) for k, v in splits.items()},
            }
            logger.info("Dataset formatting finished: split_counts=%s", artifacts["split_counts"])
            self.workspace.mark_stage_completed(StageName.DATASET_FORMATTER, artifacts=artifacts)
            return artifacts
        except Exception as exc:
            self.workspace.mark_stage_failed(StageName.DATASET_FORMATTER, str(exc))
            raise
