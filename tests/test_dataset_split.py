"""Tests for dataset validation and splitting."""

import pytest
from backend.services.dataset import dataset_service


def test_validate_lines_valid():
    lines = [
        '{"prompt": "What is 2+2?", "split": "train"}',
        '{"prompt": "What is 3*5?", "split": "val"}',
        '{"prompt": "What is 10/2?", "split": "test"}'
    ]
    valid_rows, errors = dataset_service.validate_lines(lines)
    assert len(valid_rows) == 3
    assert len(errors) == 0


def test_validate_lines_invalid():
    lines = [
        '{"prompt": ""}',  # empty prompt
        '{"question": "missing prompt"}',  # missing prompt
        'not valid json',
        '{"prompt": "valid", "split": "invalid_split"}'
    ]
    valid_rows, errors = dataset_service.validate_lines(lines)
    assert len(valid_rows) == 0
    assert len(errors) == 4


def test_auto_split_ratios():
    rows = [{"prompt": f"Query {i}"} for i in range(100)]
    split_rows = dataset_service.split_dataset(
        rows, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42
    )
    assert len(split_rows) == 100
    train_count = sum(1 for r in split_rows if r["split"] == "train")
    val_count = sum(1 for r in split_rows if r["split"] == "val")
    test_count = sum(1 for r in split_rows if r["split"] == "test")

    assert train_count == 80
    assert val_count == 10
    assert test_count == 10
