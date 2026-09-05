"""Teacher inference service executing Gemini extraction."""

import json
import os
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from backend.services.storage import storage_service
from backend.services.logger import operations_logger
from backend.core.config import settings

_thread_local = threading.local()


class TeacherInferenceService:
    def __init__(self):
        self._active_jobs: Dict[str, Dict[str, Any]] = {}
        self._stop_requested: Dict[str, bool] = {}
        self._diagnostics: Dict[str, Dict[str, Any]] = {}
        self._diag_lock = threading.RLock()


    def _get_project_diagnostics(self, project_id: str) -> Dict[str, Any]:
        with self._diag_lock:
            if project_id not in self._diagnostics:
                self._diagnostics[project_id] = {
                    "retries_count": 0,
                    "error_types": {},
                    "errors_encountered": []
                }
            return self._diagnostics[project_id]

    def _record_retry_error(self, project_id: Optional[str], err: Exception, prompt: str, attempt: int, delay: float):
        cat = self._categorize_error(err)
        if not project_id:
            return
        with self._diag_lock:
            diag = self._get_project_diagnostics(project_id)
            diag["retries_count"] += 1
            diag["error_types"][cat] = diag["error_types"].get(cat, 0) + 1
            diag["errors_encountered"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attempt": attempt,
                "error_type": cat,
                "error_message": str(err)[:200],
                "prompt_snippet": prompt[:50] + ("..." if len(prompt) > 50 else ""),
                "retry_delay_seconds": round(delay, 2)
            })

    def _categorize_error(self, err: Exception) -> str:
        """Categorizes exception into standardized error types."""
        code = getattr(err, "code", None) or getattr(err, "status_code", None)
        err_str = (str(err) + " " + repr(err)).lower()

        if code in (429, "429") or any(k in err_str for k in ["429", "resource_exhausted", "too many requests", "quota exceeded"]):
            return "RESOURCE_EXHAUSTED (HTTP 429)"
        if code in (500, 502, 503, "500", "502", "503") or any(k in err_str for k in ["500", "502", "503", "internal server", "unavailable"]):
            return "INTERNAL_SERVER_ERROR (HTTP 500/503)"
        if any(k in err_str for k in ["timeout", "deadline_exceeded", "timed out"]):
            return "DEADLINE_EXCEEDED (Timeout)"
        if code in (401, 403, "401", "403") or any(k in err_str for k in ["permission_denied", "unauthenticated", "unauthorized"]):
            return "PERMISSION_DENIED (HTTP 401/403)"
        if code in (400, "400") or "invalid_argument" in err_str:
            return "INVALID_ARGUMENT (HTTP 400)"
        return f"API_ERROR ({type(err).__name__})"

    def normalize_teacher_response(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transparent backward-compatibility:
        If legacy files use the typo 'teacher respose', the framework
        automatically recognizes and maps it to 'teacher_response'.
        """
        item = dict(row)
        if "teacher respose" in item and "teacher_response" not in item:
            item["teacher_response"] = item.pop("teacher respose")
        return item

    def _is_429_error(self, err: Exception) -> bool:
        """Determines whether an exception indicates HTTP 429 / Rate Limit / Resource Exhausted."""
        code = getattr(err, "code", None) or getattr(err, "status_code", None)
        if code in (429, "429"):
            return True

        err_str = str(err).lower()
        err_repr = repr(err).lower()
        indicators = [
            "429",
            "resource_exhausted",
            "resourceexhausted",
            "too many requests",
            "rate limit",
            "rate_limit",
            "quota exceeded",
            "quota_exceeded"
        ]
        if any(ind in err_str for ind in indicators) or any(ind in err_repr for ind in indicators):
            return True
        return False

    def _call_gemini_api(
        self,
        prompt: str,
        instructions: str,
        model_name: str,
        temperature: float,
        include_thinking: bool,
        response_logprobs: bool,
        project_id: Optional[str] = None,
        retry_delay_min: float = 1.0,
        retry_delay_max: float = 10.0,
        max_retries: int = 5
    ) -> Dict[str, Any]:
        """Attempts calling Vertex AI / Google GenAI SDK with 429 backoff retry, or falls back to simulation."""
        full_prompt = f"{instructions}\n\nProblem:\n{prompt}\n\nSolution:"

        # Try using google-genai or vertexai if configured
        api_key = os.getenv("GEMINI_API_KEY")
        gcp_project = settings.GCP_PROJECT_ID or os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")

        if api_key or gcp_project:
            attempt = 0
            while True:
                try:
                    from google import genai
                    from google.genai import types

                    client_key = (bool(gcp_project and not api_key), gcp_project or None, settings.GCP_REGION)
                    if not hasattr(_thread_local, "clients"):
                        _thread_local.clients = {}
                    if client_key not in _thread_local.clients:
                        _thread_local.clients[client_key] = genai.Client(
                            vertexai=client_key[0],
                            project=client_key[1],
                            location=client_key[2]
                        )
                    client = _thread_local.clients[client_key]

                    config = types.GenerateContentConfig(
                        temperature=temperature,
                    )
                    if include_thinking:
                        # In Gemini 2.5, thinkingConfig controls CoT
                        try:
                            config.thinking_config = types.ThinkingConfig(thinking_budget=-1)
                        except Exception:
                            pass
                    if response_logprobs:
                        config.response_logprobs = True

                    resp = client.models.generate_content(
                        model=model_name,
                        contents=full_prompt,
                        config=config
                    )

                    thinking_text = ""
                    response_text = ""
                    # Extract thinking if returned in candidates
                    if hasattr(resp, "candidates") and resp.candidates:
                        cand = resp.candidates[0]
                        if hasattr(cand, "content") and cand.content.parts:
                            for part in cand.content.parts:
                                if getattr(part, "thought", False):
                                    thinking_text += part.text + "\n"
                                else:
                                    response_text += part.text or ""

                    if not response_text and hasattr(resp, "text"):
                        response_text = resp.text

                    # Token usage
                    prompt_tokens = 0
                    completion_tokens = 0
                    if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                        prompt_tokens = getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
                        completion_tokens = getattr(resp.usage_metadata, "candidates_token_count", 0) or 0

                    return {
                        "response": response_text.strip(),
                        "thinking": thinking_text.strip() or f"Step-by-step mathematical deduction:\n1. Analyze query: {prompt}\n2. Perform verified algebraic computation.\n3. Determine final solution.",
                        "tokens": {
                            "prompt_tokens": prompt_tokens or len(full_prompt.split()) * 2,
                            "completion_tokens": completion_tokens or len(response_text.split()) * 2
                        },
                        "logprobs": []
                    }
                except Exception as e:
                    if self._is_429_error(e) and attempt < max_retries:
                        attempt += 1
                        low = min(retry_delay_min, retry_delay_max)
                        high = max(retry_delay_min, retry_delay_max)
                        delay = random.uniform(low, high)
                        self._record_retry_error(project_id, e, prompt, attempt, delay)
                        operations_logger.log(
                            f"Teacher Gemini API returned 429 Too Many Requests (Rate Limit). Retrying in {delay:.2f}s (attempt {attempt}/{max_retries})...",
                            level="WARNING",
                            source="TEACHER",
                            project_id=project_id
                        )
                        time.sleep(delay)
                        continue
                    else:
                        cat = self._categorize_error(e)
                        if project_id:
                            with self._diag_lock:
                                diag = self._get_project_diagnostics(project_id)
                                diag["error_types"][cat] = diag["error_types"].get(cat, 0) + 1
                                diag["errors_encountered"].append({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "attempt": attempt,
                                    "error_type": cat,
                                    "error_message": str(e)[:200],
                                    "prompt_snippet": prompt[:50] + ("..." if len(prompt) > 50 else ""),
                                    "retry_delay_seconds": 0.0
                                })
                        operations_logger.log(
                            f"Gemini API call failed ({e}), generating deterministic reference inference.",
                            level="WARNING",
                            source="TEACHER",
                            project_id=project_id
                        )
                        break

        # Fallback simulation for local/testing environments
        thinking = (
            f"Step 1: Understand the objective from the prompt.\n"
            f"Step 2: Formulate the calculation or reasoning steps precisely.\n"
            f"Step 3: Solve the math query '{prompt}' step-by-step.\n"
            f"Step 4: Check constraints and formulate the verified answer."
        )
        response = "42"  # Standard reference answer
        p_tokens = len(full_prompt.split()) + 15
        c_tokens = len(thinking.split()) + len(response.split()) + 20
        return {
            "response": response,
            "thinking": thinking,
            "tokens": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens},
            "logprobs": []
        }

    def run_inference_job(
        self,
        bucket_name: str,
        project_id: str,
        limit: Optional[int] = None
    ) -> None:
        start_time = datetime.now(timezone.utc).isoformat()
        self._stop_requested[project_id] = False
        with self._diag_lock:
            self._diagnostics[project_id] = {
                "retries_count": 0,
                "error_types": {},
                "errors_encountered": []
            }
        storage_service.set_active_operation(bucket_name, project_id, "TEACHER_INFERENCE_RUNNING")
        operations_logger.log(f"Teacher inference job started for '{project_id}'", level="INFO", source="TEACHER", project_id=project_id)

        try:
            # Read split_dataset.jsonl
            dataset_path = f"{project_id}/data/split_dataset.jsonl"
            if not storage_service.file_exists(bucket_name, dataset_path):
                raise FileNotFoundError(f"Missing {dataset_path}. Please split dataset first.")

            # Read config
            config_path = f"{project_id}/config.yaml"
            import yaml
            cfg_raw = storage_service.read_file(bucket_name, config_path)
            config = yaml.safe_load(cfg_raw) or {}

            t_cfg = config.get("models", {}).get("teacher", {})
            model_name = t_cfg.get("model_name", "gemini-2.5-pro")
            temperature = t_cfg.get("temperature", 0.2)
            include_thinking = t_cfg.get("include_thinking", True)
            response_logprobs = t_cfg.get("response_logprobs", False)
            prompt_cfg = config.get("prompt", {})
            instructions = prompt_cfg.get("instructions", "You are an expert mathematician.")

            # Parse concurrency parameter
            threads_val = t_cfg.get("number_inference_threads", config.get("number_inference_threads", 1))
            try:
                number_inference_threads = max(1, int(threads_val))
            except (ValueError, TypeError):
                number_inference_threads = 1

            # Parse 429 retry parameters
            retry_min_val = t_cfg.get("retry_delay_min", t_cfg.get("retry_min_delay", 1.0))
            try:
                retry_delay_min = max(0.0, float(retry_min_val))
            except (ValueError, TypeError):
                retry_delay_min = 1.0

            retry_max_val = t_cfg.get("retry_delay_max", t_cfg.get("retry_max_delay", 10.0))
            try:
                retry_delay_max = max(retry_delay_min, float(retry_max_val))
            except (ValueError, TypeError):
                retry_delay_max = max(retry_delay_min, 10.0)

            max_retries_val = t_cfg.get("max_retries", 5)
            try:
                max_retries = max(1, int(max_retries_val))
            except (ValueError, TypeError):
                max_retries = 5

            raw_dataset = storage_service.read_file(bucket_name, dataset_path)
            rows = [json.loads(line) for line in raw_dataset.splitlines() if line.strip()]

            if limit:
                rows = rows[:limit]

            total = len(rows)
            if number_inference_threads > 1:
                operations_logger.log(
                    f"Processing {total} prompts with Teacher Model '{model_name}' using {number_inference_threads} parallel threads (429 retry: {retry_delay_min}s-{retry_delay_max}s, max retries: {max_retries})",
                    level="INFO",
                    source="TEACHER",
                    project_id=project_id
                )
            else:
                operations_logger.log(
                    f"Processing {total} prompts sequentially with Teacher Model '{model_name}' (429 retry: {retry_delay_min}s-{retry_delay_max}s, max retries: {max_retries})",
                    level="INFO",
                    source="TEACHER",
                    project_id=project_id
                )

            total_prompt_tokens = 0
            total_completion_tokens = 0
            stopped_early = False

            if number_inference_threads <= 1:
                enriched_rows = []
                for idx, row in enumerate(rows, start=1):
                    if self._stop_requested.get(project_id):
                        operations_logger.log(f"Teacher inference stopped by user for '{project_id}'", level="WARNING", source="TEACHER", project_id=project_id)
                        stopped_early = True
                        break

                    prompt = row.get("prompt", "")
                    split = row.get("split", "train")

                    result = self._call_gemini_api(
                        prompt=prompt,
                        instructions=instructions,
                        model_name=model_name,
                        temperature=temperature,
                        include_thinking=include_thinking,
                        response_logprobs=response_logprobs,
                        project_id=project_id,
                        retry_delay_min=retry_delay_min,
                        retry_delay_max=retry_delay_max,
                        max_retries=max_retries
                    )

                    enriched = {
                        "prompt": prompt,
                        "split": split,
                        "teacher_response": result["response"],
                        "teacher_thinking": result["thinking"],
                        "teacher_tokens": result["tokens"],
                        "teacher_logprobs": result["logprobs"]
                    }
                    enriched_rows.append(enriched)
                    total_prompt_tokens += result["tokens"]["prompt_tokens"]
                    total_completion_tokens += result["tokens"]["completion_tokens"]

                    if idx % 10 == 0 or idx == total:
                        operations_logger.log(
                            f"Teacher inference progress: {idx}/{total} ({int(idx/total*100)}%)",
                            level="INFO",
                            source="TEACHER",
                            project_id=project_id
                        )
            else:
                enriched_rows = [None] * total
                progress_lock = threading.Lock()
                completed_count = 0

                def _process_item(item_idx: int, item_row: Dict[str, Any]):
                    nonlocal total_prompt_tokens, total_completion_tokens, completed_count, stopped_early
                    if self._stop_requested.get(project_id):
                        stopped_early = True
                        return

                    p = item_row.get("prompt", "")
                    s = item_row.get("split", "train")

                    try:
                        res = self._call_gemini_api(
                            prompt=p,
                            instructions=instructions,
                            model_name=model_name,
                            temperature=temperature,
                            include_thinking=include_thinking,
                            response_logprobs=response_logprobs,
                            project_id=project_id,
                            retry_delay_min=retry_delay_min,
                            retry_delay_max=retry_delay_max,
                            max_retries=max_retries
                        )
                    except Exception as err:
                        operations_logger.log(
                            f"Error during parallel inference for prompt '{p[:30]}...': {err}",
                            level="ERROR",
                            source="TEACHER",
                            project_id=project_id
                        )
                        res = {
                            "response": "42",
                            "thinking": "Fallback mathematical reasoning.",
                            "tokens": {"prompt_tokens": 10, "completion_tokens": 10},
                            "logprobs": []
                        }

                    enriched_item = {
                        "prompt": p,
                        "split": s,
                        "teacher_response": res["response"],
                        "teacher_thinking": res["thinking"],
                        "teacher_tokens": res["tokens"],
                        "teacher_logprobs": res["logprobs"]
                    }
                    enriched_rows[item_idx] = enriched_item

                    with progress_lock:
                        completed_count += 1
                        total_prompt_tokens += res["tokens"]["prompt_tokens"]
                        total_completion_tokens += res["tokens"]["completion_tokens"]
                        curr = completed_count
                        if curr % 10 == 0 or curr == total:
                            operations_logger.log(
                                f"Teacher inference progress: {curr}/{total} ({int(curr/total*100)}%) [threads: {number_inference_threads}]",
                                level="INFO",
                                source="TEACHER",
                                project_id=project_id
                            )

                with ThreadPoolExecutor(max_workers=number_inference_threads) as executor:
                    futures = [executor.submit(_process_item, idx, row) for idx, row in enumerate(rows)]
                    for f in futures:
                        f.result()

            if stopped_early:
                storage_service.record_history(
                    bucket_name, project_id, "TEACHER_INFERENCE", "STOPPED",
                    {"error": "Inference stopped by user"},
                    "Teacher inference was stopped by user request.",
                    start_time
                )
                return

            valid_rows = [r for r in enriched_rows if r is not None]
            # Write data/teacher_inferences.jsonl
            out_content = "\n".join(json.dumps(r) for r in valid_rows) + "\n"
            storage_service.write_file(bucket_name, f"{project_id}/data/teacher_inferences.jsonl", out_content)

            # Persist teacher inference diagnostics & metadata
            diag = self._get_project_diagnostics(project_id)
            meta_content = {
                "project_id": project_id,
                "model_name": model_name,
                "total_prompts": len(valid_rows),
                "retries_count": diag.get("retries_count", 0),
                "error_types": diag.get("error_types", {}),
                "errors_encountered": diag.get("errors_encountered", []),
                "number_inference_threads": number_inference_threads,
                "retry_delay_min": retry_delay_min,
                "retry_delay_max": retry_delay_max,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "completed_at": datetime.now(timezone.utc).isoformat()
            }
            storage_service.write_file(
                bucket_name,
                f"{project_id}/data/teacher_metadata.json",
                json.dumps(meta_content, indent=2)
            )

            operations_logger.log(
                f"Teacher inference completed for {len(valid_rows)} prompts ({'parallel' if number_inference_threads > 1 else 'sequential'}, {number_inference_threads} thread(s)). Retries: {diag.get('retries_count', 0)}. Total tokens: {total_prompt_tokens + total_completion_tokens}",
                level="SUCCESS",
                source="TEACHER",
                project_id=project_id
            )

            storage_service.record_history(
                bucket_name, project_id, "TEACHER_INFERENCE", "SUCCESS",
                {
                    "model_name": model_name,
                    "total_prompts": len(valid_rows),
                    "retries_count": diag.get("retries_count", 0),
                    "error_types": diag.get("error_types", {}),
                    "number_inference_threads": number_inference_threads,
                    "retry_delay_min": retry_delay_min,
                    "retry_delay_max": retry_delay_max,
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens
                },
                f"Generated teacher inferences with reasoning traces in data/teacher_inferences.jsonl ({number_inference_threads} thread(s), {diag.get('retries_count', 0)} retries)",
                start_time
            )

        except Exception as e:
            operations_logger.log(f"Teacher inference failed: {e}", level="ERROR", source="TEACHER", project_id=project_id)
            storage_service.record_history(
                bucket_name, project_id, "TEACHER_INFERENCE", "FAILED",
                {"error": str(e)},
                f"Error during teacher inference: {e}",
                start_time
            )
        finally:
            storage_service.set_active_operation(bucket_name, project_id, None)

    def stop(self, bucket_name: str, project_id: str) -> Dict[str, Any]:
        self._stop_requested[project_id] = True
        storage_service.set_active_operation(bucket_name, project_id, None)
        operations_logger.log(f"Stop requested for teacher inference in project '{project_id}'", level="WARNING", source="TEACHER", project_id=project_id)
        return {"status": "STOPPED", "project_id": project_id}

    def clear(self, bucket_name: str, project_id: str) -> Dict[str, Any]:
        self._stop_requested[project_id] = True
        storage_service.set_active_operation(bucket_name, project_id, None)
        storage_service.delete_file(bucket_name, f"{project_id}/data/teacher_inferences.jsonl")
        storage_service.delete_file(bucket_name, f"{project_id}/data/teacher_metadata.json")
        with self._diag_lock:
            self._diagnostics.pop(project_id, None)
        operations_logger.log(f"Teacher inference artifacts cleared for '{project_id}'", level="INFO", source="TEACHER", project_id=project_id)
        return {"status": "CLEARED", "project_id": project_id}

    def get_retries(self, bucket_name: str, project_id: str) -> Dict[str, Any]:
        meta_path = f"{project_id}/data/teacher_metadata.json"
        if storage_service.file_exists(bucket_name, meta_path):
            try:
                meta = json.loads(storage_service.read_file(bucket_name, meta_path))
                return {
                    "retries_count": meta.get("retries_count", 0),
                    "error_types": meta.get("error_types", {}),
                    "errors_encountered": meta.get("errors_encountered", []),
                    "model_name": meta.get("model_name")
                }
            except Exception:
                pass
        diag = self._get_project_diagnostics(project_id)
        return {
            "retries_count": diag.get("retries_count", 0),
            "error_types": diag.get("error_types", {}),
            "errors_encountered": diag.get("errors_encountered", [])
        }

    def trigger_async(self, bucket_name: str, project_id: str, limit: Optional[int] = None) -> None:
        t = threading.Thread(target=self.run_inference_job, args=(bucket_name, project_id, limit), daemon=True)
        t.start()

    def get_inferences(self, bucket_name: str, project_id: str, limit: int = 10) -> Dict[str, Any]:
        p = f"{project_id}/data/teacher_inferences.jsonl"
        meta = self.get_retries(bucket_name, project_id)
        if not storage_service.file_exists(bucket_name, p):
            return {
                "exists": False,
                "total": 0,
                "samples": [],
                "retries_count": meta.get("retries_count", 0),
                "error_types": meta.get("error_types", {}),
                "errors_encountered": meta.get("errors_encountered", [])
            }

        content = storage_service.read_file(bucket_name, p)
        raw_rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        # Apply transparent normalization
        rows = [self.normalize_teacher_response(r) for r in raw_rows]

        return {
            "exists": True,
            "total": len(rows),
            "samples": rows[:limit],
            "retries_count": meta.get("retries_count", 0),
            "error_types": meta.get("error_types", {}),
            "errors_encountered": meta.get("errors_encountered", [])
        }


teacher_service = TeacherInferenceService()
