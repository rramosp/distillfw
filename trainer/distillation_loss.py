try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from typing import Dict, Any, Optional

TensorType = Any if not HAS_TORCH else torch.Tensor


def compute_seq_kd_loss(
    logits: TensorType,
    labels: TensorType,
    ignore_index: int = -100
) -> TensorType:
    """
    Method 1: Sequence-Level Knowledge Distillation (SeqKD)
    Standard cross-entropy over teacher completion tokens.
    Shift logits and labels by 1 for causal language modeling.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index
    )
    return loss


def compute_cot_loss(
    logits: TensorType,
    labels: TensorType,
    think_mask: Optional[TensorType] = None,
    resp_mask: Optional[TensorType] = None,
    thinking_weight: float = 0.5,
    response_weight: float = 1.0,
    ignore_index: int = -100
) -> TensorType:
    """
    Method 2: Distilling Step-by-Step CoT
    L_CoT = lambda_think * L_think + lambda_resp * L_resp
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
    token_losses = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    token_losses = token_losses.view(shift_labels.size())

    if think_mask is not None and resp_mask is not None:
        shift_think = think_mask[..., 1:].contiguous()
        shift_resp = resp_mask[..., 1:].contiguous()

        think_denom = shift_think.sum().clamp(min=1.0)
        resp_denom = shift_resp.sum().clamp(min=1.0)

        loss_think = (token_losses * shift_think).sum() / think_denom
        loss_resp = (token_losses * shift_resp).sum() / resp_denom

        return thinking_weight * loss_think + response_weight * loss_resp

    # Fallback to standard cross-entropy if masks are omitted
    valid_mask = (shift_labels != ignore_index).float()
    return token_losses.sum() / valid_mask.sum().clamp(min=1.0)


def compute_topk_soft_kd_loss(
    student_logits: TensorType,
    teacher_topk_indices: TensorType,
    teacher_topk_logprobs: TensorType,
    labels: TensorType,
    temperature: float = 2.0,
    alpha: float = 0.3,
    ignore_index: int = -100
) -> TensorType:
    """
    Method 4: Top-k Soft Target KD
    L = (1 - alpha) * L_CE + alpha * L_KL
    """
    # 1. Standard cross entropy
    shift_logits = student_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index
    )

    # 2. Top-k KL divergence
    # Gather student logits for top-k indices
    gathered_student = torch.gather(shift_logits, -1, teacher_topk_indices[..., 1:])
    student_soft = F.log_softmax(gathered_student / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_topk_logprobs[..., 1:] / temperature, dim=-1)

    loss_kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)

    return (1.0 - alpha) * loss_ce + alpha * loss_kl


def compute_on_policy_gkd_loss(
    student_logits: TensorType,
    ref_logits: TensorType,
    student_tokens: TensorType,
    rewards: TensorType,
    beta: float = 0.1
) -> TensorType:
    """
    Method 3: Generalized Knowledge Distillation (GKD) with Teacher reward feedback (DPO-style).
    """
    shift_student = student_logits[..., :-1, :].contiguous()
    shift_ref = ref_logits[..., :-1, :].contiguous()
    shift_tokens = student_tokens[..., 1:].contiguous()

    log_p_student = torch.gather(
        F.log_softmax(shift_student, dim=-1), -1, shift_tokens.unsqueeze(-1)
    ).squeeze(-1)
    log_p_ref = torch.gather(
        F.log_softmax(shift_ref, dim=-1), -1, shift_tokens.unsqueeze(-1)
    ).squeeze(-1)

    log_ratio = log_p_student - log_p_ref
    loss = -F.logsigmoid(beta * log_ratio * rewards).mean()
    return loss
