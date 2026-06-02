from __future__ import annotations

from typing import Tuple

from transformers import AutoImageProcessor, AutoModelForImageClassification


def build_swinv2_classifier(
    num_labels: int,
    label2id: dict,
    id2label: dict,
    model_name: str = "microsoft/swinv2-large-patch4-window12-192-22k",
) -> Tuple[AutoModelForImageClassification, AutoImageProcessor]:
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    return model, image_processor
