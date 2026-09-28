"""Image + pseudo-mask dataset for PIDNet training and validation.

Geometry: every image is letterboxed to ``S x S`` with exactly the arithmetic of
``firecls.baselines.preprocessing.Letterbox`` (the evaluation transform), and its mask with the
same scale and offsets using nearest-neighbour resampling. Letterboxing rather than centre-cropping
keeps the full field of view, so no annotated fire or smoke is cut away.

Augmentation (training only), after letterboxing: one operation drawn uniformly from the ten
listed by Pesonen et al. (crop, vertical flip, rotation, perspective, erasing, grayscale, blur,
inversion, sharpness, colour jitter), then an independent horizontal flip with p = 0.5. Geometric
operations transform image and mask jointly via ``torchvision.tv_tensors``; photometric ones
leave the mask untouched; erasing zeroes the mask inside the erased rectangle, since the target
there is no longer visible.

Edge targets follow upstream PIDNet ``gen_sample``: Canny on the label map, a 6-pixel border
suppressed, dilation with a 4x4 kernel. They are computed after augmentation, so edges always
match the augmented mask.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision import tv_tensors
from torchvision.transforms import v2

from firecls.baselines.pidnet.masks import decode_mask, mask_path, read_mask
from firecls.baselines.preprocessing import Letterbox

EDGE_KERNEL = 4  # upstream edge_size
EDGE_BORDER = 6  # upstream x_k_size / y_k_size


def letterbox_geometry(width: int, height: int, size: int) -> tuple[int, int, int, int]:
    """``(new_w, new_h, left, top)`` -- identical to ``preprocessing.Letterbox``."""
    ratio = size / max(width, height)
    new_w, new_h = max(1, round(width * ratio)), max(1, round(height * ratio))
    return new_w, new_h, (size - new_w) // 2, (size - new_h) // 2


def letterbox_mask(encoded: np.ndarray, size: int) -> np.ndarray:
    """Encoded uint8 ``[H, W]`` mask -> ``[size, size]`` with nearest resampling and zero padding."""
    height, width = encoded.shape
    new_w, new_h, left, top = letterbox_geometry(width, height, size)
    resized = np.asarray(Image.fromarray(encoded).resize((new_w, new_h), Image.NEAREST), dtype=np.uint8)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def mask_edges(mask: np.ndarray) -> np.ndarray:
    """Binary boundary target ``[H, W]`` (float32) from a ``[2, H, W]`` multi-label mask."""
    edge = np.zeros(mask.shape[1:], dtype=np.uint8)
    for channel in mask:
        edge |= cv2.Canny(np.ascontiguousarray(channel, dtype=np.uint8) * 255, 0.1, 0.2)
    if EDGE_BORDER and min(edge.shape) > 2 * EDGE_BORDER:
        edge[:EDGE_BORDER], edge[-EDGE_BORDER:] = 0, 0
        edge[:, :EDGE_BORDER], edge[:, -EDGE_BORDER:] = 0, 0
    edge = cv2.dilate(edge, np.ones((EDGE_KERNEL, EDGE_KERNEL), np.uint8), iterations=1)
    return (edge > 50).astype(np.float32)


def _joint_erasing(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    i, j, h, w, _ = T.RandomErasing.get_params(image, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=[0.0])
    image, mask = image.clone(), mask.clone()
    image[..., i : i + h, j : j + w] = 0.0
    mask[..., i : i + h, j : j + w] = 0
    return tv_tensors.Image(image), tv_tensors.Mask(mask)


class JointAugment:
    """Single uniformly drawn augmentation + independent 50 % horizontal flip."""

    def __init__(self, size: int, pad_value: int = 114) -> None:
        fill = {tv_tensors.Image: pad_value / 255.0, tv_tensors.Mask: 0}
        self.ops: list[tuple[str, Callable]] = [
            ("crop", v2.RandomResizedCrop(size, scale=(0.25, 1.0), antialias=True)),
            ("vflip", v2.RandomVerticalFlip(p=1.0)),
            ("rotation", v2.RandomRotation(degrees=30, fill=fill)),
            ("perspective", v2.RandomPerspective(distortion_scale=0.5, p=1.0, fill=fill)),
            ("erasing", _joint_erasing),
            ("grayscale", v2.Grayscale(num_output_channels=3)),
            ("blur", v2.GaussianBlur(kernel_size=9, sigma=(0.1, 3.0))),
            ("invert", v2.RandomInvert(p=1.0)),
            ("sharpness", v2.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0)),
            ("colour_jitter", v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)),
        ]
        self.hflip = v2.RandomHorizontalFlip(p=1.0)

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.ops]

    def __call__(self, image: torch.Tensor, mask: torch.Tensor, op: int | None = None):
        """``image`` float ``[3,S,S]``, ``mask`` uint8 ``[2,S,S]``. ``op`` forces an operation (tests)."""
        image, mask = tv_tensors.Image(image), tv_tensors.Mask(mask)
        index = int(torch.randint(len(self.ops), ())) if op is None else op
        image, mask = self.ops[index][1](image, mask)
        if torch.rand(()) < 0.5:
            image, mask = self.hflip(image, mask)
        # Not in-place: Grayscale returns an expanded (memory-sharing) view.
        return image.as_subclass(torch.Tensor).clamp(0.0, 1.0), mask.as_subclass(torch.Tensor)


class PseudoMaskDataset(Dataset):
    """Returns ``(image [3,S,S] in [0,1], mask [2,S,S] float, edge [S,S] float, label)``.

    Images are *not* normalised here: normalisation lives inside ``PidnetPresenceScores`` so the
    exported ONNX graph and this dataset agree on the ``[0, 1]`` letterboxed input.
    """

    def __init__(
        self,
        samples: Sequence[tuple[Path, int]],
        domain: str,
        masks_root: Path,
        img_size: int,
        augment: bool = False,
        pad_value: int = 114,
    ) -> None:
        self.samples = list(samples)
        self.domain = domain
        self.masks_root = Path(masks_root)
        self.img_size = int(img_size)
        self.letterbox = Letterbox(self.img_size, pad_value)
        self.augment = JointAugment(self.img_size, pad_value) if augment else None

    def __len__(self) -> int:
        return len(self.samples)

    def mask_file(self, index: int) -> Path:
        return mask_path(self.masks_root, self.domain, self.samples[index][0])

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        path = self.mask_file(index)
        if not path.exists():
            raise FileNotFoundError(f"Pseudo-mask {path} missing; run scripts/pidnet/generate_sam_masks.py first.")
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        encoded = read_mask(path)
        if encoded.shape != (image.height, image.width):
            raise ValueError(f"{path}: mask {encoded.shape} does not match image {(image.height, image.width)}")
        image_t = self.letterbox(image)
        mask_t = torch.from_numpy(decode_mask(letterbox_mask(encoded, self.img_size)).astype(np.uint8))
        if self.augment is not None:
            image_t, mask_t = self.augment(image_t, mask_t)
        mask_np = mask_t.numpy()
        edge = torch.from_numpy(mask_edges(mask_np))
        return image_t.contiguous(), mask_t.float(), edge, int(label)
