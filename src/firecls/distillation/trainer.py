from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from firecls.distillation.losses import MultiTeacherDistillationLoss
from firecls.utils import accuracy_from_logits


@dataclass
class DistillationMetrics:
    loss: float
    hard_loss: float
    soft_loss: float
    accuracy: float


class MultiTeacherDistillationTrainer:
    def __init__(
        self,
        student_model: nn.Module,
        teacher_models: Dict[str, nn.Module],
        temperature: float = 4.0,
        alpha: float = 0.3,
        beta: float = 0.7,
        teacher_weights: Iterable[float] | None = None,
        aggregation: str = "weighted_avg",
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student = student_model.to(self.device)
        self.teachers = self._prepare_teachers(teacher_models)

        self.distillation_loss = MultiTeacherDistillationLoss(
            num_teachers=len(self.teachers),
            alpha=alpha,
            beta=beta,
            temperature=temperature,
            teacher_weights=teacher_weights,
            aggregation=aggregation,
        ).to(self.device)

    def _prepare_teachers(self, teacher_models: Dict[str, nn.Module]) -> Dict[str, nn.Module]:
        teachers = {}
        for domain, teacher in teacher_models.items():
            teacher = teacher.to(self.device)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
            teachers[domain] = teacher
        return teachers

    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> DistillationMetrics:
        self.student.train()
        total_loss = 0.0
        total_hard = 0.0
        total_soft = 0.0
        total_acc = 0.0
        total = 0

        progress = tqdm(loader, desc=f"Epoch {epoch}")
        for images, labels in progress:
            images = images.to(self.device)
            labels = labels.to(self.device)

            student_outputs = self.student(images)
            student_logits = getattr(student_outputs, "logits", student_outputs)

            teacher_logits_list = []
            with torch.no_grad():
                for teacher in self.teachers.values():
                    teacher_outputs = teacher(images)
                    teacher_logits = getattr(teacher_outputs, "logits", teacher_outputs)
                    teacher_logits_list.append(teacher_logits)

            loss, hard_loss, soft_loss = self.distillation_loss(
                student_logits, teacher_logits_list, labels
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_hard += hard_loss.item() * batch_size
            total_soft += soft_loss.item() * batch_size
            total_acc += accuracy_from_logits(student_logits, labels) * batch_size
            total += batch_size

            progress.set_postfix(
                loss=total_loss / max(1, total),
                hard=total_hard / max(1, total),
                soft=total_soft / max(1, total),
                acc=total_acc / max(1, total),
            )

        return DistillationMetrics(
            loss=total_loss / max(1, total),
            hard_loss=total_hard / max(1, total),
            soft_loss=total_soft / max(1, total),
            accuracy=total_acc / max(1, total),
        )

    def evaluate(self, loader: DataLoader) -> float:
        self.student.eval()
        total_acc = 0.0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.student(images)
                logits = getattr(outputs, "logits", outputs)
                total_acc += accuracy_from_logits(logits, labels) * labels.size(0)
                total += labels.size(0)
        return total_acc / max(1, total)

    def evaluate_domains(self, loaders: Dict[str, DataLoader]) -> Dict[str, float]:
        return {name: self.evaluate(loader) for name, loader in loaders.items()}
