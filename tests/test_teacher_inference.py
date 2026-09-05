"""Tests for teacher inference service and backward compatibility."""

import pytest
from backend.services.teacher import teacher_service


def test_typo_normalization():
    # Specification Section 3:
    # "Transparent backward-compatibility: If legacy files use the typo 'teacher respose',
    # the framework automatically recognizes and maps it to 'teacher_response'."
    legacy_row = {
        "prompt": "What is 15 * 18?",
        "split": "train",
        "teacher respose": "270",
        "teacher_thinking": "Multiply 15 by 18"
    }
    normalized = teacher_service.normalize_teacher_response(legacy_row)
    assert "teacher_response" in normalized
    assert normalized["teacher_response"] == "270"
    assert "teacher respose" not in normalized


def test_already_normalized_row():
    row = {
        "prompt": "What is 15 * 18?",
        "split": "train",
        "teacher_response": "270"
    }
    normalized = teacher_service.normalize_teacher_response(row)
    assert normalized["teacher_response"] == "270"
