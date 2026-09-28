"""Pseudo-mask files and the Segment Anything box-prompt teacher.

Masks are stored as one single-channel uint8 PNG per image, at the image's original resolution,
with ``bit0 = fire`` and ``bit1 = smoke``. Two bits in one lossless file keep the multi-label
target (a pixel may be both, e.g. flames seen through smoke) at a quarter of the disk cost of
two separate masks, and a PNG can be inspected with any image viewer.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from firecls.baselines.boxes import Box

FIRE_BIT = 1
SMOKE_BIT = 2
CHANNEL_BITS = {"fire": FIRE_BIT, "smoke": SMOKE_BIT}


def encode_mask(fire: np.ndarray, smoke: np.ndarray) -> np.ndarray:
    """Two boolean ``[H, W]`` masks -> one uint8 ``[H, W]`` array."""
    if fire.shape != smoke.shape:
        raise ValueError(f"Mask shapes differ: {fire.shape} vs {smoke.shape}")
    return (fire.astype(bool) * FIRE_BIT + smoke.astype(bool) * SMOKE_BIT).astype(np.uint8)


def decode_mask(encoded: np.ndarray) -> np.ndarray:
    """uint8 ``[H, W]`` -> bool ``[2, H, W]`` ordered (fire, smoke)."""
    return np.stack([(encoded & FIRE_BIT) > 0, (encoded & SMOKE_BIT) > 0])


def mask_path(masks_root: Path, domain: str, image_path: Path | str) -> Path:
    """``<masks_root>/<domain>/<image stem>.png``; stems are unique within a FASDD domain."""
    stem = Path(str(image_path).replace("\\", "/")).stem  # tolerate Windows paths from the CSVs
    return Path(masks_root) / domain / f"{stem}.png"


def write_mask(path: Path, encoded: np.ndarray) -> None:
    """Atomic write, so an interrupted run never leaves a truncated PNG that resume would skip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.png")
    Image.fromarray(np.ascontiguousarray(encoded, dtype=np.uint8)).save(tmp, format="PNG")
    os.replace(tmp, path)


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.uint8).copy()


def box_to_pixels(box: Box, width: int, height: int) -> np.ndarray:
    """Normalised ``Box`` -> ``[x1, y1, x2, y2]`` in pixel coordinates (SAM's box prompt format)."""
    return np.array([box.x1 * width, box.y1 * height, box.x2 * width, box.y2 * height], dtype=np.float32)


def filled_box(shape: tuple[int, int], box_px: np.ndarray) -> np.ndarray:
    height, width = shape
    x1, y1 = int(np.floor(box_px[0])), int(np.floor(box_px[1]))
    x2, y2 = int(np.ceil(box_px[2])), int(np.ceil(box_px[3]))
    mask = np.zeros(shape, dtype=bool)
    mask[max(0, y1) : min(height, max(y2, y1 + 1)), max(0, x1) : min(width, max(x2, x1 + 1))] = True
    return mask


class BoxSegmenter(Protocol):
    def set_image(self, image: np.ndarray) -> None: ...

    def predict_box(self, box_xyxy: np.ndarray) -> np.ndarray: ...


class SamBoxSegmenter:
    """Zero-shot SAM teacher: one box prompt per ground-truth box, ``multimask_output=False``."""

    def __init__(self, checkpoint: Path, model_type: str = "vit_h", device: str = "cuda") -> None:
        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device).eval()
        self.predictor = SamPredictor(sam)

    def set_image(self, image: np.ndarray) -> None:
        self.predictor.set_image(image, image_format="RGB")

    def predict_box(self, box_xyxy: np.ndarray) -> np.ndarray:
        masks, _, _ = self.predictor.predict(box=box_xyxy.astype(np.float32), multimask_output=False)
        return masks[0].astype(bool)


def pseudo_mask(segmenter: BoxSegmenter, image: np.ndarray, boxes: Sequence[Box]) -> tuple[np.ndarray, dict]:
    """Union of SAM masks per class for one RGB ``[H, W, 3]`` image.

    If SAM raises, or returns an empty mask for a box, the filled box is used for that box and
    counted, so a teacher failure degrades to box supervision instead of a silent false negative.
    """
    height, width = image.shape[:2]
    channels = {"fire": np.zeros((height, width), dtype=bool), "smoke": np.zeros((height, width), dtype=bool)}
    stats = {"boxes": 0, "fallback_error": 0, "fallback_empty": 0}
    if not boxes:
        return encode_mask(channels["fire"], channels["smoke"]), stats
    try:
        segmenter.set_image(image)
        image_ok = True
    except Exception:
        image_ok = False
    for box in boxes:
        stats["boxes"] += 1
        box_px = box_to_pixels(box, width, height)
        segment = None
        if image_ok:
            try:
                segment = np.asarray(segmenter.predict_box(box_px), dtype=bool)
                if segment.shape != (height, width):
                    raise ValueError(f"SAM mask shape {segment.shape} != image {(height, width)}")
            except Exception:
                segment = None
        if segment is None:
            stats["fallback_error"] += 1
            segment = filled_box((height, width), box_px)
        elif not segment.any():
            stats["fallback_empty"] += 1
            segment = filled_box((height, width), box_px)
        channels[box.cls] |= segment
    return encode_mask(channels["fire"], channels["smoke"]), stats


def generate_masks(
    samples: dict[str, list[tuple[Path, str]]],
    boxes: dict[str, dict[str, list[Box]]],
    segmenter: BoxSegmenter,
    masks_root: Path,
    limit: int | None = None,
    progress=None,
) -> dict:
    """Write one pseudo-mask per ``(image_path, csv_label)`` in ``samples[domain]``; resumable.

    Existing PNGs are skipped. ``limit`` caps the number of *newly written* masks in this call.
    Per-domain counters are returned for the manifest; label/box disagreements are counted, not
    fixed, because the CSV label (from TDML) is the image-level ground truth for every method.
    """
    from firecls.baselines.boxes import label_from_boxes

    stats: dict[str, dict] = {}
    written_total = 0
    for domain, items in samples.items():
        s = stats.setdefault(
            domain,
            {"images": len(items), "written": 0, "skipped_existing": 0, "no_boxes": 0, "missing_box_entry": 0,
             "label_mismatch": 0, "boxes": 0, "fallback_error": 0, "fallback_empty": 0, "seconds": 0.0},
        )
        domain_boxes = boxes.get(domain, {})
        for image_path, label in items:
            if progress is not None:
                progress.update(1)
            out = mask_path(masks_root, domain, image_path)
            if out.exists():
                s["skipped_existing"] += 1
                continue
            if limit is not None and written_total >= limit:
                break
            start = time.perf_counter()
            stem = Path(image_path).stem
            image_boxes = domain_boxes.get(stem)
            if image_boxes is None:
                s["missing_box_entry"] += 1
                image_boxes = []
            if not image_boxes:
                s["no_boxes"] += 1
            if label_from_boxes(image_boxes) != label:
                s["label_mismatch"] += 1
            with Image.open(image_path) as raw:
                rgb = np.asarray(raw.convert("RGB"), dtype=np.uint8)
            encoded, box_stats = pseudo_mask(segmenter, rgb, image_boxes)
            write_mask(out, encoded)
            for key, value in box_stats.items():
                s[key] += value
            s["written"] += 1
            s["seconds"] += time.perf_counter() - start
            written_total += 1
    return stats
