from __future__ import annotations

from torchvision import transforms


def build_full_resize_transform(img_size: int, mean, std):
    """USTC_SmokeRS secondary preprocessing (external/USTC_PROTOCOL.md, Section 4): the whole image is resized
    directly to img_size x img_size, with no larger resize and no centre crop."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_eval_transform(img_size: int, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
