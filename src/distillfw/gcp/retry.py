"""Retry helper with strictly increasing exponential backoff for Vertex AI / Gemini API requests."""

from __future__ import annotations

import random
import time
from typing import Any, Callable, TypeVar

from distillfw.logging_utils import get_logger

T = TypeVar("T")


def compute_increasing_backoff_delays(
    max_retries: int = 12,
    initial_delay: float = 2.0,
    max_delay: float = 120.0,
    multiplier: float = 1.8,
    jitter_ratio: float = 0.1,
) -> list[float]:
    """Compute a strictly increasing sequence of `max_retries` wait durations (in seconds).

    Guarantees `len(delays) == max_retries` (with `max_retries >= 10`) and
    `delays[0] < delays[1] < ... < delays[-1]` so every retry waits longer than the last.
    """
    if max_retries < 10:
        max_retries = 10

    delays: list[float] = []
    prev_delay = 0.0
    for attempt_idx in range(max_retries):
        raw = min(max_delay, initial_delay * (multiplier ** attempt_idx))
        jitter = raw * jitter_ratio * random.random()
        candidate = raw + jitter
        # Enforce strict monotonic increase between consecutive retries
        min_increment = max(0.5, initial_delay * 0.25)
        delay = max(candidate, prev_delay + min_increment)
        delays.append(round(delay, 3))
        prev_delay = delay
    return delays


def call_with_exponential_backoff(
    fn: Callable[[], T],
    *,
    max_retries: int = 12,
    initial_delay: float = 2.0,
    max_delay: float = 120.0,
    multiplier: float = 1.8,
    operation_name: str = "Gemini API request",
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    """Execute `fn()` with at least 10 retries and strictly increasing wait intervals.

    Retries on transient / quota errors (e.g., 429 RESOURCE_EXHAUSTED, 503 UNAVAILABLE,
    OVERLOADED_TOO_MANY_RETRIES_PER_REQUEST, connection timeouts). Non-retryable
    configuration errors (e.g., invalid argument / unsupported logprobs) are raised immediately.
    """
    effective_retries = max(10, max_retries)
    delays = compute_increasing_backoff_delays(
        max_retries=effective_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        multiplier=multiplier,
    )
    logger = get_logger("gcp.retry")

    last_exc: Exception | None = None
    for attempt in range(effective_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            err_msg = str(exc)
            # Do not retry permanent 400 INVALID_ARGUMENT, 403 PERMISSION_DENIED, or 404 NOT_FOUND errors
            is_permanent_invalid_arg = (
                "INVALID_ARGUMENT" in err_msg
                or "Logprobs is not supported" in err_msg
                or "404 NOT_FOUND" in err_msg
                or "was not found or your project does not have access to it" in err_msg
                or "PERMISSION_DENIED" in err_msg
            )
            if is_permanent_invalid_arg or attempt >= effective_retries:
                raise

            wait_seconds = delays[attempt]
            logger.warning(
                "%s failed (retry %d/%d): %s | Waiting %.2fs before next attempt...",
                operation_name,
                attempt + 1,
                effective_retries,
                err_msg,
                wait_seconds,
            )
            sleep_fn(wait_seconds)

    assert last_exc is not None
    raise last_exc
