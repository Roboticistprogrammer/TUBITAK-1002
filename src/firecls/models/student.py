from __future__ import annotations

from typing import List

from firecls.config import DEFAULT_MODEL_NAME
from firecls.models.swinv2 import build_swinv2_classifier


def build_student_model(
    classes: List[str],
    model_name: str | None = None,
):
    model, image_processor = build_swinv2_classifier(
        num_labels=len(classes),
        label2id={name: i for i, name in enumerate(classes)},
        id2label={i: name for i, name in enumerate(classes)},
        model_name=model_name or DEFAULT_MODEL_NAME,
    )
    return model, image_processor
