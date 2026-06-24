from __future__ import annotations

from pathlib import Path

import torch

from firecls.config import DEFAULT_CLASSES, DEFAULT_MODEL_NAME
from firecls.models.swinv2 import build_swinv2_classifier


def load_checkpoint_model(checkpoint_path: Path, device: torch.device | str = "cpu"):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    classes = checkpoint.get("classes", DEFAULT_CLASSES)
    model_name = checkpoint.get("model_name", DEFAULT_MODEL_NAME)
    model, image_processor = build_swinv2_classifier(
        num_labels=len(classes),
        label2id={name: index for index, name in enumerate(classes)},
        id2label={index: name for index, name in enumerate(classes)},
        model_name=model_name,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    metadata = {
        "epoch": checkpoint.get("epoch"),
        "avg_val": checkpoint.get("avg_val"),
        "classes": classes,
        "model_name": model_name,
    }
    return model, image_processor, metadata
