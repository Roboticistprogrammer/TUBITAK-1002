"""Original training recipe of every model, and the fine-tuning recipe derived from it (PLUME_PROTOCOL.md, Section 7).

Each model keeps its original optimizer, augmentation, batch size and AMP setting. Fine-tuning runs 5 epochs at one tenth of the
original learning rate, with any learning-rate schedule restarted over those 5 epochs.

Where each value comes from:
* students (KD uniform / routed): `scripts/train_student.py`: AdamW lr 3e-5, weight decay 0.01, constant lr, gradient clipping
  1.0, no AMP. The batch size was **8**, not the script default 16: the training logs show 7,666 iterations per epoch, which is
  61,323 training images / 8.
* no-KD student: `scripts/cnn/train_classifier.py --arch swinv2_base_nokd`: the same optimizer, plus AMP; batch **8** (logs: 7,665
  iterations per epoch, drop_last).
* MobileNetV3-Large and ResNet-50: `scripts/cnn/train_classifier.py`: AdamW lr 3e-4, weight decay 0.05, cosine schedule with 1
  warm-up epoch, clipping 1.0, AMP, batch 16 (all recorded in their `history.json`).
* RS teacher: `scripts/train_teacher.py`: AdamW lr 3e-5, weight decay 0.01, constant lr, no gradient clipping. Its batch size and
  AMP setting were NOT recorded anywhere; the script default batch 16 is used, with AMP (see the protocol, Deviations).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

FINETUNE_EPOCHS = 5
LR_FACTOR = 0.1


@dataclass(frozen=True)
class Recipe:
    lr: float
    weight_decay: float
    schedule: str  # "constant" or "cosine"
    warmup_epochs: int
    grad_clip: float | None
    amp: bool
    batch_size: int
    drop_last: bool

    def asdict(self) -> dict:
        return asdict(self)


ORIGINAL = {
    "teacher": Recipe(3e-5, 0.01, "constant", 0, None, True, 16, False),
    "kd": Recipe(3e-5, 0.01, "constant", 0, 1.0, False, 8, False),
    "nokd": Recipe(3e-5, 0.01, "constant", 0, 1.0, True, 8, True),
    "cnn": Recipe(3e-4, 0.05, "cosine", 1, 1.0, True, 16, True),
}

KINDS = {  # CLI kind -> (recipe key, is a SwinV2 checkpoint, trained on all three domains, loss)
    "teacher": ("teacher", True, False, "ce"),
    "kd_routed": ("kd", True, True, "kd"),
    "kd_uniform": ("kd", True, True, "kd"),
    "nokd": ("nokd", True, True, "ce"),
    "mobilenet_v3_large": ("cnn", False, True, "ce"),
    "resnet50": ("cnn", False, True, "ce"),
}


def finetune_recipe(kind: str, lr_factor: float = LR_FACTOR, batch_size: int | None = None) -> Recipe:
    base = ORIGINAL[KINDS[kind][0]]
    return replace(base, lr=base.lr * lr_factor, batch_size=batch_size or base.batch_size)


def lr_factor_at(recipe: Recipe, step: int, steps_per_epoch: int, epochs: int) -> float:
    """Learning-rate multiplier at an optimizer step; the schedule starts at step 0 of the fine-tuning (restarted)."""
    if recipe.schedule == "constant":
        return 1.0
    import math

    warmup = recipe.warmup_epochs * steps_per_epoch
    total = max(1, epochs * steps_per_epoch)
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
