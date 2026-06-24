from __future__ import annotations

from torchvision import transforms


def build_eval_transform(img_size: int, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
