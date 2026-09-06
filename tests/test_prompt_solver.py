"""Unit tests for multi-domain prompt synthesis and question solving."""

import pytest
from backend.services.deployment import _synthesize_solution, _safe_eval_math_ast


def test_safe_eval_math_ast():
    assert _safe_eval_math_ast("15 * 18") == 270.0
    assert _safe_eval_math_ast("25 * 14") == 350.0
    assert _safe_eval_math_ast("100 / 4") == 25.0
    assert _safe_eval_math_ast("2 ** 8") == 256.0
    assert _safe_eval_math_ast("__import__('os').system('ls')") is None


def test_arithmetic_natural_language_operators():
    ans, thinking = _synthesize_solution("What is 25 multiplied by 14?")
    assert ans == "350"
    assert "350" in thinking

    ans, _ = _synthesize_solution("What is 144 divided by 12?")
    assert ans == "12"

    ans, _ = _synthesize_solution("Calculate 85 plus 115")
    assert ans == "200"

    ans, _ = _synthesize_solution("What is 100 minus 37?")
    assert ans == "63"


def test_powers_and_roots():
    ans, _ = _synthesize_solution("What is the square root of 625?")
    assert ans == "25"

    ans, _ = _synthesize_solution("Calculate 2 to the power of 8")
    assert ans == "256"


def test_percentages():
    ans, _ = _synthesize_solution("What is 20 percent of 150?")
    assert ans == "30"


def test_physics_kinematics():
    ans, _ = _synthesize_solution("If a car travels at 60 mph for 3.5 hours, how far does it go?")
    assert ans == "210"


def test_domain_knowledge_geography_and_science():
    ans, _ = _synthesize_solution("What is the capital of France?")
    assert ans == "Paris"

    ans, _ = _synthesize_solution("What is the capital of Japan?")
    assert ans == "Tokyo"

    ans, _ = _synthesize_solution("What is the speed of light?")
    assert "299,792,458" in ans

    ans, _ = _synthesize_solution("Who wrote Hamlet?")
    assert ans == "William Shakespeare"


def test_domain_knowledge_ml_and_systems():
    ans, _ = _synthesize_solution("What does LoRA stand for in machine learning?")
    assert "Low-Rank Adaptation" in ans

    ans, _ = _synthesize_solution("Explain vLLM serving engine")
    assert "PagedAttention" in ans

    ans, _ = _synthesize_solution("What is knowledge distillation?")
    assert "Teacher" in ans


def test_general_query_decomposition():
    ans, thinking = _synthesize_solution("Can you explain quantum supremacy experimental benchmarks?")
    assert ans != "42"
    assert "quantum" in ans.lower() or "supremacy" in ans.lower()
    assert "Query Decomposition" in thinking
