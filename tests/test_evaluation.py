"""Tests for evaluation metrics computation."""

from backend.services.evaluation import compute_rouge_scores, compute_bleu_and_em


def test_rouge_and_exact_match():
    predictions = ["The answer is 42", "Paris is the capital of France", "x = 10"]
    references = ["The answer is 42", "Paris is the capital of France", "x = 12"]

    em_bleu = compute_bleu_and_em(predictions, references)
    assert em_bleu["exact_match"] == 66.67  # 2 out of 3 match exactly
    assert em_bleu["json_compliance_rate"] == 100.0

    rouge = compute_rouge_scores(predictions, references)
    assert rouge["rouge1"] > 70.0
    assert rouge["rougeL"] > 70.0
