"""Tests for PyTorch off-policy and gray-box distillation loss functions."""

from __future__ import annotations

import torch

from distillfw.algorithms.losses import (
    distillm2_contrastive_loss,
    forward_kl_loss,
    jensen_shannon_loss,
    reverse_kl_loss,
    skew_kl_loss,
    sparse_topk_kl_loss,
)


def test_distillation_losses_zero_when_identical_and_differentiable() -> None:
    torch.manual_seed(0)
    teacher_logits = torch.randn(2, 6, 32)
    student_same = teacher_logits.clone().requires_grad_(True)
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]], dtype=torch.float32)

    assert torch.allclose(forward_kl_loss(student_same, teacher_logits, mask), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(reverse_kl_loss(student_same, teacher_logits, mask), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(jensen_shannon_loss(student_same, teacher_logits, mask), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(skew_kl_loss(student_same, teacher_logits, mask), torch.tensor(0.0), atol=1e-5)

    student_diff = torch.randn(2, 6, 32, requires_grad=True)
    loss = distillm2_contrastive_loss(
        student_diff, teacher_logits, student_diff, teacher_logits, mask, mask
    )
    assert loss.item() > 0.0
    loss.backward()
    assert student_diff.grad is not None
    assert torch.isfinite(student_diff.grad).all()


def test_sparse_topk_kl_loss_from_api_logprobs() -> None:
    torch.manual_seed(1)
    student_logits = torch.randn(2, 5, 50, requires_grad=True)
    teacher_logits = torch.randn(2, 5, 50)
    topk_probs, topk_ids = torch.softmax(teacher_logits, dim=-1).topk(5, dim=-1)
    mask = torch.ones(2, 5)

    for div in ("forward_kl", "reverse_kl", "skew_kl"):
        loss = sparse_topk_kl_loss(
            student_logits=student_logits,
            topk_token_ids=topk_ids,
            topk_logprobs=topk_probs.log(),
            mask=mask,
            divergence=div,
        )
        assert loss.item() >= 0.0
        loss.backward()
        assert student_logits.grad is not None
        student_logits.grad.zero_()
