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


def test_is_429_detection():
    class MockAPIError(Exception):
        def __init__(self, code, msg):
            self.code = code
            super().__init__(msg)

    assert teacher_service._is_429_error(MockAPIError(429, "Too Many Requests"))
    assert teacher_service._is_429_error(Exception("429 Resource has been exhausted (e.g. check quota)"))
    assert teacher_service._is_429_error(Exception("RESOURCE_EXHAUSTED: quota exceeded for quota metric"))
    assert teacher_service._is_429_error(Exception("rate limit exceeded"))
    assert not teacher_service._is_429_error(Exception("404 Not Found"))
    assert not teacher_service._is_429_error(ValueError("Invalid argument"))


def test_429_retry_with_delay(monkeypatch):
    import time
    delays = []

    def mock_sleep(d):
        delays.append(d)

    monkeypatch.setattr("backend.services.teacher.time.sleep", mock_sleep)

    call_count = 0

    class MockGenAIClient:
        class models:
            @staticmethod
            def generate_content(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    class Error429(Exception):
                        code = 429
                    raise Error429("Rate limit exceeded")
                
                class MockResp:
                    text = "42"
                    candidates = []
                    usage_metadata = None
                return MockResp()

    import backend.services.teacher as teacher_mod
    monkeypatch.setattr("backend.services.teacher.settings.GCP_PROJECT_ID", "mock-project")
    teacher_mod._thread_local.clients = {
        (True, "mock-project", "us-central1"): MockGenAIClient()
    }

    result = teacher_service._call_gemini_api(
        prompt="Calculate 6 * 7",
        instructions="Calculate answer",
        model_name="gemini-2.5-pro",
        temperature=0.2,
        include_thinking=True,
        response_logprobs=False,
        project_id="test-proj",
        retry_delay_min=1.0,
        retry_delay_max=5.0,
        max_retries=5
    )

    assert call_count == 3
    retry_delays = [d for d in delays if d >= 1.0]
    assert len(retry_delays) == 2
    for d in retry_delays:
        assert 1.0 <= d <= 5.0
    assert result["response"] == "42"


def test_parallel_inference_preserves_order(tmp_path, monkeypatch):
    from backend.services.storage import storage_service
    monkeypatch.setattr("backend.core.config.settings.LOCAL_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr("backend.core.config.settings.STORAGE_MODE", "local")
    storage_service.use_gcs = False

    bucket = "test-bucket"
    proj = "test-project"

    # Setup config with number_inference_threads = 4
    config_yaml = """
models:
  teacher:
    model_name: "gemini-2.5-pro"
    number_inference_threads: 4
    retry_delay_min: 0.1
    retry_delay_max: 0.2
"""
    storage_service.write_file(bucket, f"{proj}/config.yaml", config_yaml)

    # Setup 20 distinct prompts
    lines = [f'{{"prompt": "Problem {i}", "split": "train"}}' for i in range(20)]
    storage_service.write_file(bucket, f"{proj}/data/split_dataset.jsonl", "\n".join(lines))

    # Run inference job
    teacher_service.run_inference_job(bucket, proj)

    # Verify inferences file
    inferences = teacher_service.get_inferences(bucket, proj, limit=50)
    assert inferences["exists"] is True
    assert inferences["total"] == 20

    # Ensure ordering is preserved exactly from 0 to 19
    for i, sample in enumerate(inferences["samples"]):
        assert sample["prompt"] == f"Problem {i}"


def test_sequential_inference(tmp_path, monkeypatch):
    from backend.services.storage import storage_service
    monkeypatch.setattr("backend.core.config.settings.LOCAL_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr("backend.core.config.settings.STORAGE_MODE", "local")
    storage_service.use_gcs = False

    bucket = "test-bucket"
    proj = "test-project"

    # Setup config with number_inference_threads = 1 (sequential)
    config_yaml = """
models:
  teacher:
    model_name: "gemini-2.5-pro"
    number_inference_threads: 1
    retry_delay_min: 0.1
    retry_delay_max: 0.2
"""
    storage_service.write_file(bucket, f"{proj}/config.yaml", config_yaml)

    lines = [f'{{"prompt": "Seq Problem {i}", "split": "train"}}' for i in range(5)]
    storage_service.write_file(bucket, f"{proj}/data/split_dataset.jsonl", "\n".join(lines))

    teacher_service.run_inference_job(bucket, proj)

    inferences = teacher_service.get_inferences(bucket, proj, limit=10)
    assert inferences["exists"] is True
    assert inferences["total"] == 5
    for i, sample in enumerate(inferences["samples"]):
        assert sample["prompt"] == f"Seq Problem {i}"


def test_teacher_config_validation():
    from backend.core.models import TeacherModelConfig
    from pydantic import ValidationError

    cfg = TeacherModelConfig(number_inference_threads=4, retry_delay_min=2.0, retry_delay_max=8.0)
    assert cfg.number_inference_threads == 4
    assert cfg.retry_delay_min == 2.0
    assert cfg.retry_delay_max == 8.0

    # Test validator: number_inference_threads must be >= 1
    with pytest.raises(ValidationError):
        TeacherModelConfig(number_inference_threads=0)

    with pytest.raises(ValidationError):
        TeacherModelConfig(number_inference_threads=-2)

