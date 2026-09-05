"""Tests for master configuration loading and schema validation."""

import os
import yaml
from backend.core.models import MasterConfig


def test_load_sample_config():
    with open("sample_config.yaml", "r") as f:
        data = yaml.safe_load(f)

    cfg = MasterConfig(**data)
    assert cfg.project.id == "distill-gemma-math-v1"
    assert cfg.models.teacher.model_name == "gemini-2.5-pro"
    assert cfg.models.teacher.number_inference_threads == 4
    assert cfg.models.teacher.retry_delay_min == 1.0
    assert cfg.models.teacher.retry_delay_max == 10.0
    assert cfg.models.teacher.max_retries == 5
    assert cfg.models.student.model_name_or_path == "google/gemma-2-9b"
    assert cfg.distillation.method == "cot_distillation"
    assert cfg.distillation.cot_weights.thinking_weight == 0.5
    assert cfg.training.hardware.accelerator_type == "NVIDIA_L4"
    assert cfg.deployment.serving_framework == "vllm"


def test_default_config_validity():
    cfg = MasterConfig()
    dumped = cfg.model_dump()
    assert dumped["models"]["teacher"]["model_name"] == "gemini-2.5-pro"
    assert dumped["training"]["hardware"]["accelerator_type"] == "NVIDIA_L4"
