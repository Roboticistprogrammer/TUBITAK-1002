from __future__ import annotations

import torch

from bootstrap import setup_path

setup_path()

from firecls.config import DEFAULT_CLASSES, DEFAULT_IMG_SIZE, DEFAULT_MODEL_NAME
from firecls.models.swinv2 import build_swinv2_classifier


def main() -> None:
    model, _ = build_swinv2_classifier(
        num_labels=len(DEFAULT_CLASSES),
        label2id={name: i for i, name in enumerate(DEFAULT_CLASSES)},
        id2label={i: name for i, name in enumerate(DEFAULT_CLASSES)},
        model_name=DEFAULT_MODEL_NAME,
    )

    dummy = torch.randn(2, 3, DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE)
    with torch.no_grad():
        logits = model(dummy).logits

    print(f"Smoke test OK. Logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
