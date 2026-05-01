from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch

from firecls.config import DEFAULT_CLASSES, DEFAULT_MODEL_NAME
from firecls.models.swinv2 import build_swinv2_classifier


def load_teacher_checkpoint(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> Tuple[torch.nn.Module, List[str], str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes = checkpoint.get("classes", DEFAULT_CLASSES)
    model_name = checkpoint.get("model_name", DEFAULT_MODEL_NAME)

    model, _ = build_swinv2_classifier(
        num_labels=len(classes),
        label2id={name: i for i, name in enumerate(classes)},
        id2label={i: name for i, name in enumerate(classes)},
        model_name=model_name,
    )
    model.load_state_dict(checkpoint["model"])

    if device is not None:
        model = model.to(device)
    model.eval()
    return model, classes, model_name


def load_teachers(
    teacher_paths: Dict[str, Path],
    device: torch.device | None = None,
) -> Tuple[Dict[str, torch.nn.Module], List[str], str]:
    teachers: Dict[str, torch.nn.Module] = {}
    classes: List[str] | None = None
    model_name: str | None = None

    for domain, path in teacher_paths.items():
        teacher, checkpoint_classes, checkpoint_model = load_teacher_checkpoint(path, device=device)
        teachers[domain] = teacher
        if classes is None:
            classes = checkpoint_classes
        if model_name is None:
            model_name = checkpoint_model

    if classes is None or model_name is None:
        raise ValueError("No teacher checkpoints were loaded.")

    return teachers, classes, model_name
