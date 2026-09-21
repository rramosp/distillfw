"""Pytest configuration and shared fixtures for distillfw test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_hf_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure a default HF_TOKEN is set for unit tests unless explicitly overridden."""
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_default")
