"""PyTorch implementations of SOTA off-policy and gray-box LLM distillation losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(loss_per_token: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Compute token-level masked mean over active (non-padding, non-prompt) tokens."""
    if mask is None:
        return loss_per_token.mean()
    active = mask.to(loss_per_token.dtype)
    denom = active.sum().clamp_min(1.0)
    return (loss_per_token * active).sum() / denom


def forward_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Standard Word-Level Forward KL Divergence: KL(p_teacher || q_student).

    Mode-covering objective (Hinton et al., 2015).
    """
    t = temperature
    log_p = F.log_softmax(teacher_logits / t, dim=-1)
    log_q = F.log_softmax(student_logits / t, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1) * (t**2)
    return _masked_mean(kl, mask)


def reverse_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Reverse KL Divergence: KL(q_student || p_teacher).

    Mode-seeking objective preventing student mass in low-density teacher regions (MiniLLM).
    """
    t = temperature
    log_p = F.log_softmax(teacher_logits / t, dim=-1)
    log_q = F.log_softmax(student_logits / t, dim=-1)
    q = log_q.exp()
    rkl = (q * (log_q - log_p)).sum(dim=-1) * (t**2)
    return _masked_mean(rkl, mask)


def jensen_shannon_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    beta: float = 0.5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generalized Jensen-Shannon Divergence: JSD_beta(p || q) (Agarwal et al., GKD)."""
    t = temperature
    log_p = F.log_softmax(teacher_logits / t, dim=-1)
    log_q = F.log_softmax(student_logits / t, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    m = beta * p + (1.0 - beta) * q
    log_m = m.clamp_min(1e-12).log()
    jsd = (beta * (p * (log_p - log_m)).sum(dim=-1) + (1.0 - beta) * (q * (log_q - log_m)).sum(dim=-1)) * (t**2)
    return _masked_mean(jsd, mask)


def skew_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    alpha: float = 0.1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Skew KL Divergence from DistiLLM (Ko et al., ICML 2024).

    Computes KL(p || alpha * p + (1 - alpha) * q) for bounded, stable gradients.
    """
    t = temperature
    log_p = F.log_softmax(teacher_logits / t, dim=-1)
    log_q = F.log_softmax(student_logits / t, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    skew_dist = (alpha * p + (1.0 - alpha) * q).clamp_min(1e-12)
    skl = (p * (log_p - skew_dist.log())).sum(dim=-1) * (t**2)
    return _masked_mean(skl, mask)


def distillm2_contrastive_loss(
    student_logits_on_teacher: torch.Tensor,
    teacher_logits_on_teacher: torch.Tensor,
    student_logits_on_student: torch.Tensor,
    teacher_logits_on_student: torch.Tensor,
    mask_teacher: torch.Tensor | None = None,
    mask_student: torch.Tensor | None = None,
    alpha: float = 0.1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Contrastive Asymmetric Loss from DistiLLM-2 (Ko et al., ICML 2025).

    Applies Skew KLD on teacher trajectories (pulling up teacher modes)
    and Reverse KLD on offline student trajectories (pushing down student errors).
    """
    loss_pos = skew_kl_loss(
        student_logits=student_logits_on_teacher,
        teacher_logits=teacher_logits_on_teacher,
        mask=mask_teacher,
        alpha=alpha,
        temperature=temperature,
    )
    loss_neg = reverse_kl_loss(
        student_logits=student_logits_on_student,
        teacher_logits=teacher_logits_on_student,
        mask=mask_student,
        temperature=temperature,
    )
    return 0.5 * (loss_pos + loss_neg)


def sparse_topk_kl_loss(
    student_logits: torch.Tensor,
    topk_token_ids: torch.Tensor,
    topk_logprobs: torch.Tensor,
    mask: torch.Tensor | None = None,
    divergence: str = "forward_kl",
    skew_alpha: float = 0.1,
    tail_eps: float = 1e-7,
) -> torch.Tensor:
    """Gray-box API Distillation Loss using Gemini Top-K `response_logprobs`.

    Reconstructs a full-vocabulary teacher distribution from top-k token IDs and logprobs,
    distributing residual probability mass `(1 - sum(exp(topk_logprobs)))` uniformly across
    the remaining vocabulary items, then computes Forward KL, Reverse KL, or Skew KL.
    """
    vocab_size = student_logits.size(-1)
    log_q = F.log_softmax(student_logits, dim=-1)
    q = log_q.exp()

    # Reconstruct full teacher probability tensor p of shape [batch, seq_len, vocab_size]
    topk_probs = topk_logprobs.exp().clamp(min=0.0, max=1.0)
    topk_mass = topk_probs.sum(dim=-1, keepdim=True).clamp(max=1.0)
    residual_mass = (1.0 - topk_mass).clamp_min(tail_eps)
    unseen_count = max(vocab_size - topk_token_ids.size(-1), 1)
    uniform_tail_prob = residual_mass / unseen_count

    p = uniform_tail_prob.expand_as(student_logits).clone()
    p.scatter_(dim=-1, index=topk_token_ids, src=topk_probs)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    log_p = p.clamp_min(1e-12).log()

    if divergence == "forward_kl":
        token_loss = (p * (log_p - log_q)).sum(dim=-1)
    elif divergence == "reverse_kl":
        token_loss = (q * (log_q - log_p)).sum(dim=-1)
    elif divergence == "skew_kl":
        skew_dist = (skew_alpha * p + (1.0 - skew_alpha) * q).clamp_min(1e-12)
        token_loss = (p * (log_p - skew_dist.log())).sum(dim=-1)
    else:
        raise ValueError(f"Unsupported divergence type: {divergence!r}")

    return _masked_mean(token_loss, mask)
