from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F


class DistillationLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        temperature: float = 4.0,
        distillation_type: str = "soft",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.distillation_type = distillation_type
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        hard_loss = self.ce_loss(student_logits, labels)
        if self.distillation_type == "hard":
            return hard_loss

        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (self.temperature**2)

        if self.distillation_type == "soft":
            return soft_loss
        return self.alpha * hard_loss + self.beta * soft_loss


class MultiTeacherDistillationLoss(nn.Module):
    def __init__(
        self,
        num_teachers: int,
        alpha: float = 0.3,
        beta: float = 0.7,
        temperature: float = 4.0,
        teacher_weights: Optional[Iterable[float]] = None,
        aggregation: str = "weighted_avg",
    ) -> None:
        super().__init__()
        self.num_teachers = num_teachers
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.aggregation = aggregation
        self.ce_loss = nn.CrossEntropyLoss()

        if teacher_weights is None:
            teacher_weights = [1.0 / num_teachers] * num_teachers
        weights = torch.tensor(list(teacher_weights), dtype=torch.float32)
        self.teacher_weights = nn.Parameter(weights, requires_grad=False)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits_list: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hard_loss = self.ce_loss(student_logits, labels)
        aggregated_teacher_soft = self._aggregate_teacher_soft(teacher_logits_list)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)

        soft_loss = (
            F.kl_div(student_soft, aggregated_teacher_soft, reduction="batchmean") * (self.temperature**2)
        )
        total_loss = self.alpha * hard_loss + self.beta * soft_loss
        return total_loss, hard_loss, soft_loss

    def _aggregate_teacher_soft(self, teacher_logits_list: List[torch.Tensor]) -> torch.Tensor:
        if len(teacher_logits_list) != self.num_teachers:
            raise ValueError(
                f"Expected {self.num_teachers} teacher logits, got {len(teacher_logits_list)}."
            )

        if self.aggregation == "weighted_avg":
            aggregated = torch.zeros_like(
                F.softmax(teacher_logits_list[0] / self.temperature, dim=1)
            )
            for index, teacher_logits in enumerate(teacher_logits_list):
                teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
                aggregated += self.teacher_weights[index] * teacher_soft
            return aggregated

        if self.aggregation == "max":
            teacher_softs = [
                F.softmax(logits / self.temperature, dim=1) for logits in teacher_logits_list
            ]
            return torch.max(torch.stack(teacher_softs), dim=0)[0]

        if self.aggregation == "uncertainty_weighted":
            weights = []
            for teacher_logits in teacher_logits_list:
                teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
                confidence = teacher_soft.max(dim=1)[0]
                weights.append(confidence)

            weights_tensor = torch.stack(weights, dim=0)
            weights_tensor = F.softmax(weights_tensor, dim=0)

            aggregated = torch.zeros_like(
                F.softmax(teacher_logits_list[0] / self.temperature, dim=1)
            )
            for index, teacher_logits in enumerate(teacher_logits_list):
                teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
                aggregated += weights_tensor[index].unsqueeze(1) * teacher_soft
            return aggregated

        raise ValueError(f"Unknown aggregation strategy: {self.aggregation}")
