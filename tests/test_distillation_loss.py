import pytest
torch = pytest.importorskip("torch")
from trainer.distillation_loss import (
    compute_seq_kd_loss,
    compute_cot_loss,
    compute_on_policy_gkd_loss,
    compute_topk_soft_kd_loss
)


def test_seq_kd_loss():
    batch_size = 2
    seq_len = 8
    vocab_size = 100

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Mask out first 3 tokens as prompt
    labels[:, :3] = -100

    loss = compute_seq_kd_loss(logits, labels)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_cot_loss():
    batch_size = 2
    seq_len = 8
    vocab_size = 100

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    think_mask = torch.zeros(batch_size, seq_len)
    think_mask[:, 1:4] = 1.0

    resp_mask = torch.zeros(batch_size, seq_len)
    resp_mask[:, 4:] = 1.0

    loss = compute_cot_loss(
        logits=logits,
        labels=labels,
        think_mask=think_mask,
        resp_mask=resp_mask,
        thinking_weight=0.5,
        response_weight=1.0
    )
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_topk_soft_kd_loss():
    batch_size = 2
    seq_len = 8
    vocab_size = 50
    k = 5

    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_topk_indices = torch.randint(0, vocab_size, (batch_size, seq_len, k))
    teacher_topk_logprobs = torch.randn(batch_size, seq_len, k)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = compute_topk_soft_kd_loss(
        student_logits=student_logits,
        teacher_topk_indices=teacher_topk_indices,
        teacher_topk_logprobs=teacher_topk_logprobs,
        labels=labels,
        temperature=2.0,
        alpha=0.3
    )
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
