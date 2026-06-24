from __future__ import annotations

from typing import Tuple

from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification


def build_swinv2_classifier(
    num_labels: int,
    label2id: dict,
    id2label: dict,
    model_name: str = "microsoft/swinv2-large-patch4-window12-192-22k",
    pretrained: bool = True,
) -> Tuple[AutoModelForImageClassification, AutoImageProcessor]:
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model_kwargs = {
        "num_labels": num_labels,
        "label2id": label2id,
        "id2label": id2label,
    }
    if pretrained:
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True,
            **model_kwargs,
        )
    else:
        config = AutoConfig.from_pretrained(model_name, **model_kwargs)
        model = AutoModelForImageClassification.from_config(config)
    return model, image_processor
