"""Convolutional image classifiers from torchvision (ResNet-50, MobileNetV3-Large).

These are the CNN backbones cited in the literature review (He et al., 2016; Howard et al.,
2019). Because the proposed model is an image-level classifier, the like-for-like CNN baseline
is the same backbone with a 4-way classification head, trained on the same labels, splits,
augmentation and input geometry as the distilled student. Wrapping the backbones in Faster
R-CNN would change the task (boxes instead of image labels) and needs TensorRT plugins for
RoIAlign and NMS, so it is deliberately not used here (see docs/baselines/CNN_CLASSIFIERS.md).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models

from firecls.baselines.preprocessing import IMAGENET_MEAN, IMAGENET_STD, resize_center_crop_spec
from firecls.baselines.protocol import CLASSES, CLASSIFIER_IMG_SIZE, CLASSIFIER_RESIZE
from firecls.baselines.registry import LoadedBaseline, register_family

ARCHITECTURES = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
}


def build_cnn(arch: str, num_classes: int = len(CLASSES), pretrained: bool = True) -> nn.Module:
    """ImageNet-initialised backbone with its final linear layer replaced by a 4-way head."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown CNN '{arch}'. Choose from {sorted(ARCHITECTURES)}")
    constructor, weights = ARCHITECTURES[arch]
    model = constructor(weights=weights if pretrained else None)
    if arch == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:  # mobilenet_v3_large: classifier = [Linear, Hardswish, Dropout, Linear]
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def preprocessing(img_size: int = CLASSIFIER_IMG_SIZE) -> dict:
    resize = CLASSIFIER_RESIZE if img_size == CLASSIFIER_IMG_SIZE else img_size + 32
    return resize_center_crop_spec(img_size, resize, IMAGENET_MEAN, IMAGENET_STD)


@register_family("cnn")
def load(checkpoint: Path, device) -> LoadedBaseline:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("classes") != CLASSES:
        raise ValueError(f"Checkpoint class order {payload.get('classes')} != protocol {CLASSES}")
    model = build_cnn(payload["arch"], len(CLASSES), pretrained=False)
    model.load_state_dict(payload["model"])
    metadata = {
        "classes": CLASSES,
        "model_name": f"torchvision/{payload['arch']}",
        "arch": payload["arch"],
        "source": "official",
        "epoch": payload.get("epoch"),
        "avg_val": payload.get("avg_val"),
        "train_seed": payload.get("seed"),
        "recipe": payload.get("recipe"),
    }
    return LoadedBaseline(model, preprocessing(payload.get("img_size", CLASSIFIER_IMG_SIZE)), metadata)
