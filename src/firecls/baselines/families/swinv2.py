"""Hugging Face SwinV2 checkpoints (teachers, the distilled student, or a no-KD SwinV2 run).

Registering the student here lets it be scored by exactly the same evaluator and benchmark
as every baseline, which removes one source of doubt from the comparison table.
"""
from __future__ import annotations

from pathlib import Path

import torch

from firecls.baselines.preprocessing import resize_center_crop_spec
from firecls.baselines.protocol import CLASSES, CLASSIFIER_IMG_SIZE, CLASSIFIER_RESIZE
from firecls.baselines.registry import LoadedBaseline, register_family
from firecls.deployment.model import load_checkpoint_model


class LogitsOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values)
        return getattr(outputs, "logits", outputs)


@register_family("swinv2")
def load(checkpoint: Path, device) -> LoadedBaseline:
    model, processor, metadata = load_checkpoint_model(checkpoint, "cpu")
    if metadata["classes"] != CLASSES:
        raise ValueError(f"Class order {metadata['classes']} differs from protocol {CLASSES}")
    preprocessing = resize_center_crop_spec(
        CLASSIFIER_IMG_SIZE, CLASSIFIER_RESIZE, processor.image_mean, processor.image_std
    )
    return LoadedBaseline(LogitsOnly(model), preprocessing, metadata)
