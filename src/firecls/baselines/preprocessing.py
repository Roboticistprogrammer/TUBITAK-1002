"""Preprocessing described by data, so ONNX/TensorRT manifests fully specify their inputs.

Two modes cover every baseline:

* ``resize_center_crop`` -- the student's evaluation transform (classifiers, PIDNet).
* ``letterbox`` -- aspect-preserving resize with grey padding, as used by YOLO models.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resize_center_crop_spec(
    crop: int, resize: int | None = None, mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD
) -> dict:
    return {
        "mode": "resize_center_crop",
        "resize": int(resize if resize is not None else crop + 32),
        "crop": int(crop),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
    }


def letterbox_spec(size: int, pad_value: int = 114) -> dict:
    return {"mode": "letterbox", "size": int(size), "pad_value": int(pad_value), "scale": "1/255"}


def input_size(spec: dict) -> int:
    return int(spec["crop"] if spec["mode"] == "resize_center_crop" else spec["size"])


class Letterbox:
    """Resize the long side to ``size`` and pad to a square, returning a float CHW tensor in [0, 1]."""

    def __init__(self, size: int, pad_value: int = 114) -> None:
        self.size = size
        self.pad_value = pad_value

    def __call__(self, image: Image.Image) -> torch.Tensor:
        width, height = image.size
        ratio = self.size / max(width, height)
        new_w, new_h = max(1, round(width * ratio)), max(1, round(height * ratio))
        resized = image.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (self.pad_value,) * 3)
        canvas.paste(resized, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1).copy())


def build_transform(spec: dict):
    mode = spec["mode"]
    if mode == "resize_center_crop":
        return transforms.Compose(
            [
                transforms.Resize(spec["resize"]),
                transforms.CenterCrop(spec["crop"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=spec["mean"], std=spec["std"]),
            ]
        )
    if mode == "letterbox":
        return Letterbox(spec["size"], spec.get("pad_value", 114))
    raise ValueError(f"Unknown preprocessing mode: {mode}")
