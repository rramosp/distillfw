"""Custom Hugging Face Trainer subclasses for off-policy logit and contrastive distillation."""

from __future__ import annotations

from typing import Any

import torch
from transformers import Trainer

from distillfw.algorithms.losses import (
    distillm2_contrastive_loss,
    forward_kl_loss,
    jensen_shannon_loss,
    reverse_kl_loss,
    skew_kl_loss,
    sparse_topk_kl_loss,
)


class OffPolicyLogitDistillationTrainer(Trainer):
    """Single-node Trainer supporting Forward KL, Reverse KL, JSD, Skew KL, and Sparse Top-K KD."""

    def __init__(
        self,
        algorithm: str = "skew_kl",
        temperature: float = 1.0,
        skew_alpha: float = 0.1,
        jsd_beta: float = 0.5,
        teacher_model: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.algorithm = algorithm
        self.temperature = temperature
        self.skew_alpha = skew_alpha
        self.jsd_beta = jsd_beta
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.get("labels")
        mask = (labels != -100) if labels is not None else inputs.get("attention_mask")

        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        student_logits = student_outputs.logits

        # Gray-box API mode: use pre-recorded Gemini top-k logprobs from dataset
        if "topk_token_ids" in inputs and "topk_logprobs" in inputs:
            loss = sparse_topk_kl_loss(
                student_logits=student_logits,
                topk_token_ids=inputs["topk_token_ids"],
                topk_logprobs=inputs["topk_logprobs"],
                mask=mask,
                divergence=self.algorithm if self.algorithm in {"forward_kl", "reverse_kl", "skew_kl"} else "forward_kl",
                skew_alpha=self.skew_alpha,
            )
            return (loss, student_outputs) if return_outputs else loss

        # White-box mode: compute or retrieve full teacher logits
        if "teacher_logits" in inputs:
            teacher_logits = inputs["teacher_logits"]
        elif self.teacher_model is not None:
            with torch.no_grad():
                teacher_logits = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                ).logits
        else:
            # Fallback to standard cross-entropy if no teacher logits are provided
            loss = student_outputs.loss
            return (loss, student_outputs) if return_outputs else loss

        if self.algorithm == "forward_kl":
            loss = forward_kl_loss(student_logits, teacher_logits, mask=mask, temperature=self.temperature)
        elif self.algorithm == "reverse_kl":
            loss = reverse_kl_loss(student_logits, teacher_logits, mask=mask, temperature=self.temperature)
        elif self.algorithm == "jsd":
            loss = jensen_shannon_loss(
                student_logits, teacher_logits, mask=mask, beta=self.jsd_beta, temperature=self.temperature
            )
        elif self.algorithm == "skew_kl":
            loss = skew_kl_loss(
                student_logits, teacher_logits, mask=mask, alpha=self.skew_alpha, temperature=self.temperature
            )
        else:
            raise ValueError(f"Unsupported logit distillation algorithm: {self.algorithm!r}")

        return (loss, student_outputs) if return_outputs else loss


class DistiLLM2Trainer(Trainer):
    """Single-node Trainer implementing DistiLLM-2 contrastive asymmetric distillation."""

    def __init__(
        self,
        teacher_model: torch.nn.Module,
        skew_alpha: float = 0.1,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.skew_alpha = skew_alpha
        self.temperature = temperature
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        # Teacher trajectories (chosen / positive)
        t_input_ids = inputs["chosen_input_ids"]
        t_mask = inputs.get("chosen_attention_mask")
        # Student offline trajectories (rejected / negative)
        s_input_ids = inputs["rejected_input_ids"]
        s_mask = inputs.get("rejected_attention_mask")

        student_on_teacher = model(input_ids=t_input_ids, attention_mask=t_mask)
        student_on_student = model(input_ids=s_input_ids, attention_mask=s_mask)

        with torch.no_grad():
            teacher_on_teacher = self.teacher_model(input_ids=t_input_ids, attention_mask=t_mask).logits
            teacher_on_student = self.teacher_model(input_ids=s_input_ids, attention_mask=s_mask).logits

        loss = distillm2_contrastive_loss(
            student_logits_on_teacher=student_on_teacher.logits,
            teacher_logits_on_teacher=teacher_on_teacher,
            student_logits_on_student=student_on_student.logits,
            teacher_logits_on_student=teacher_on_student,
            mask_teacher=t_mask,
            mask_student=s_mask,
            alpha=self.skew_alpha,
            temperature=self.temperature,
        )
        return (loss, student_on_teacher) if return_outputs else loss
