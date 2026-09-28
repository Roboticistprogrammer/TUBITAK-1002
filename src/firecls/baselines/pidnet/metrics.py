"""Pseudo-label IoU on the validation split (a training diagnostic, not a thesis result).

Pesonen et al. report mIoU as the *mean of the sample-wise Jaccard index* for a binary
smoke/background task. We report the same statistic per channel, over the images whose
pseudo-mask contains that class (an image without the class has an undefined sample IoU when
the prediction is also empty); false positives on negative images are already penalised by
the image-level accuracy logged next to it. The dataset-level (pooled) IoU is logged as well.
"""
from __future__ import annotations

import numpy as np
import torch

from firecls.baselines.pidnet.model import SEG_CHANNELS


class SegmentationMeter:
    def __init__(self, threshold: float = 0.5, channels: list[str] = SEG_CHANNELS) -> None:
        self.threshold = threshold
        self.channels = list(channels)
        n = len(self.channels)
        self.sample_iou_sum = np.zeros(n)
        self.sample_count = np.zeros(n, dtype=np.int64)
        self.intersection = np.zeros(n)
        self.union = np.zeros(n)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        """``logits`` and ``mask`` are ``[B, C, H, W]`` at the same resolution."""
        pred = torch.sigmoid(logits.float()) > self.threshold
        target = mask > 0.5
        inter = (pred & target).flatten(2).sum(2).double()  # [B, C]
        union = (pred | target).flatten(2).sum(2).double()
        present = target.flatten(2).any(2)
        sample_iou = torch.where(present, inter / union.clamp(min=1), torch.zeros_like(inter))
        self.sample_iou_sum += sample_iou.sum(0).cpu().numpy()
        self.sample_count += present.sum(0).cpu().numpy()
        self.intersection += inter.sum(0).cpu().numpy()
        self.union += union.sum(0).cpu().numpy()

    def compute(self) -> dict:
        sample = np.where(self.sample_count > 0, self.sample_iou_sum / np.maximum(self.sample_count, 1), np.nan)
        pooled = np.where(self.union > 0, self.intersection / np.maximum(self.union, 1), np.nan)
        result = {}
        for i, name in enumerate(self.channels):
            result[f"iou_{name}"] = _float(sample[i])
            result[f"pooled_iou_{name}"] = _float(pooled[i])
            result[f"images_with_{name}"] = int(self.sample_count[i])
        result["miou"] = _float(np.nanmean(sample)) if np.isfinite(sample).any() else None
        result["pooled_miou"] = _float(np.nanmean(pooled)) if np.isfinite(pooled).any() else None
        return result


def _float(value) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)
