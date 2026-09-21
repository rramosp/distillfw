"""Stage 1: Resumable dataset generation from Gemini teacher models."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from distillfw.state import StageName, TaskWorkspace


class DatasetGenerator:
    """Stage 1: Queries Gemini teacher models and writes sharded datasets to GCS.

    Stateless & Resumable:
    - Reads prompts strictly from `<task_uri>/00_inputs/`.
    - Writes completed shards to `<task_uri>/01_raw_dataset/shard_XXXX.jsonl`.
    - Updates `progress_cursor["completed_shards"]` in `task_state.json` after each shard,
      allowing instant resumption from the exact unfinished shard if interrupted.
    """

    def __init__(
        self,
        workspace: TaskWorkspace,
        teacher_callable: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.workspace = workspace
        self.teacher_callable = teacher_callable

    def _load_prompts(self, input_uri: str) -> list[dict[str, Any]]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_file = self.workspace.storage.download_file(
                input_uri, Path(tmp_dir) / Path(input_uri).name
            )
            suffix = local_file.suffix.lower()
            if suffix == ".jsonl":
                rows = [
                    json.loads(line)
                    for line in local_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            elif suffix == ".parquet":
                rows = pd.read_parquet(local_file).to_dict(orient="records")
            elif suffix == ".csv":
                rows = pd.read_csv(local_file).to_dict(orient="records")
            else:
                raise ValueError(f"Unsupported prompt file format: {suffix}")
        return rows

    def _call_gemini_single(
        self,
        prompt_record: dict[str, Any],
        teacher_cfg: Any,
        gcp_cfg: Any,
    ) -> dict[str, Any]:
        prompt_text = prompt_record.get("prompt") or prompt_record.get("input") or str(prompt_record)

        if self.teacher_callable is not None:
            result = self.teacher_callable(prompt_text, prompt_record)
            return {
                "prompt": prompt_text,
                "completion": result["completion"],
                "thought": result.get("thought"),
                "candidates": result.get("candidates", [result["completion"]]),
                "topk_logprobs": result.get("topk_logprobs"),
                "teacher_model": teacher_cfg.model_id,
                "metadata": prompt_record.get("metadata", {}),
            }

        from google import genai
        from google.genai import types

        client = genai.Client(vertexai=True, project=gcp_cfg.project_id, location=gcp_cfg.location)

        gen_config_kwargs: dict[str, Any] = {
            "temperature": teacher_cfg.temperature,
            "top_p": teacher_cfg.top_p,
            "max_output_tokens": teacher_cfg.max_output_tokens,
            "candidate_count": teacher_cfg.candidate_count,
        }
        if teacher_cfg.system_instruction:
            gen_config_kwargs["system_instruction"] = teacher_cfg.system_instruction
        if teacher_cfg.response_logprobs:
            gen_config_kwargs["response_logprobs"] = True
            gen_config_kwargs["logprobs"] = teacher_cfg.logprobs_top_k
        if teacher_cfg.thinking_budget is not None:
            gen_config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=teacher_cfg.thinking_budget
            )

        response = client.models.generate_content(
            model=teacher_cfg.model_id,
            contents=prompt_text,
            config=types.GenerateContentConfig(**gen_config_kwargs),
        )

        thought_text: str | None = None
        answer_text: str = response.text or ""
        candidates: list[str] = []
        extracted_logprobs: list[dict[str, Any]] | None = None

        if getattr(response, "candidates", None):
            for cand in response.candidates:
                if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                    cand_parts = []
                    for part in cand.content.parts:
                        if getattr(part, "thought", False):
                            thought_text = part.text
                        elif getattr(part, "text", None):
                            cand_parts.append(part.text)
                    candidates.append("".join(cand_parts))
            # Extract top-k token logprobs if requested
            first_cand = response.candidates[0]
            if teacher_cfg.response_logprobs and getattr(first_cand, "logprobs_result", None):
                extracted_logprobs = []
                for step in getattr(first_cand.logprobs_result, "top_candidates", []) or []:
                    step_candidates = [
                        {"token": c.token, "log_probability": c.log_probability}
                        for c in getattr(step, "candidates", [])
                    ]
                    extracted_logprobs.append({"top_candidates": step_candidates})

        return {
            "prompt": prompt_text,
            "completion": answer_text,
            "thought": thought_text,
            "candidates": candidates or [answer_text],
            "topk_logprobs": extracted_logprobs,
            "teacher_model": teacher_cfg.model_id,
            "metadata": prompt_record.get("metadata", {}),
        }

    def run(self) -> dict[str, Any]:
        """Execute Stage 1 with automatic shard-level GCS checkpointing and resumption."""
        config = self.workspace.load_config()
        state = self.workspace.load_state()
        stage_rec = state.stages[StageName.DATASET_GENERATOR]

        input_uri = stage_rec.artifacts.get("input_prompts_uri")
        if not input_uri:
            input_files = self.workspace.storage.list_uris(self.workspace.inputs_dir_uri)
            if not input_files:
                raise FileNotFoundError(
                    f"No input prompt files found in {self.workspace.inputs_dir_uri}"
                )
            input_uri = input_files[0]

        completed_shards: list[int] = list(
            stage_rec.progress_cursor.get("completed_shards", [])
        )
        shard_uris: list[str] = list(stage_rec.artifacts.get("shard_uris", []))

        self.workspace.mark_stage_running(
            StageName.DATASET_GENERATOR,
            cursor_updates={"completed_shards": completed_shards},
        )

        try:
            all_prompts = self._load_prompts(input_uri)
            shard_size = config.teacher.shard_size
            total_shards = (len(all_prompts) + shard_size - 1) // shard_size

            for shard_idx in range(total_shards):
                if shard_idx in completed_shards:
                    continue

                shard_prompts = all_prompts[
                    shard_idx * shard_size : (shard_idx + 1) * shard_size
                ]
                shard_records = [
                    self._call_gemini_single(p, config.teacher, config.gcp)
                    for p in shard_prompts
                ]

                shard_uri = f"{self.workspace.raw_dataset_dir_uri}/shard_{shard_idx:04d}.jsonl"
                payload = "\n".join(json.dumps(r) for r in shard_records) + "\n"
                self.workspace.storage.write_text(shard_uri, payload)

                completed_shards.append(shard_idx)
                if shard_uri not in shard_uris:
                    shard_uris.append(shard_uri)

                self.workspace.update_stage_cursor(
                    StageName.DATASET_GENERATOR,
                    {
                        "completed_shards": completed_shards,
                        "total_shards": total_shards,
                        "total_prompts_processed": len(completed_shards) * shard_size,
                    },
                )

            artifacts = {
                "input_prompts_uri": input_uri,
                "raw_dataset_dir_uri": self.workspace.raw_dataset_dir_uri,
                "shard_uris": shard_uris,
                "num_records": len(all_prompts),
            }
            self.workspace.mark_stage_completed(StageName.DATASET_GENERATOR, artifacts=artifacts)
            return artifacts
        except Exception as exc:
            self.workspace.mark_stage_failed(StageName.DATASET_GENERATOR, str(exc))
            raise
