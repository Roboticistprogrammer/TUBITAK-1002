from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from bootstrap import setup_path

setup_path()

from firecls.config import DEFAULT_CLASSES, DEFAULT_IMG_SIZE, DEFAULT_MODEL_NAME
from firecls.models.swinv2 import build_swinv2_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    return parser.parse_args()


def build_eval_transform(img_size: int, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def main() -> None:
    args = parse_args()

    model, image_processor = build_swinv2_classifier(
        num_labels=len(DEFAULT_CLASSES),
        label2id={name: i for i, name in enumerate(DEFAULT_CLASSES)},
        id2label={i: name for i, name in enumerate(DEFAULT_CLASSES)},
        model_name=DEFAULT_MODEL_NAME,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transform = build_eval_transform(args.img_size, image_processor.image_mean, image_processor.image_std)
    image = Image.open(args.image).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor).logits
        probs = torch.softmax(logits, dim=1).squeeze(0)

    topk = min(args.topk, probs.numel())
    scores, indices = torch.topk(probs, k=topk)

    for score, idx in zip(scores.tolist(), indices.tolist()):
        print(f"{DEFAULT_CLASSES[idx]}: {score:.4f}")


if __name__ == "__main__":
    main()
