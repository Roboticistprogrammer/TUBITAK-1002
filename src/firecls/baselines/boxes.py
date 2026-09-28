"""Read FASDD bounding boxes from either of the formats FASDD distributes.

FASDD ships the same boxes as TDML, COCO, VOC and YOLO annotations. Detection- and
segmentation-based baselines need boxes, while the image-level split (train/val/test) must
come from the committed ``data_index`` CSVs so every method sees identical images. This module
therefore only returns boxes keyed by image *stem*; callers join them with the CSV split.

Boxes are returned normalised to ``[0, 1]`` as ``(x1, y1, x2, y2)`` with a canonical lower-case
class name (``"fire"`` or ``"smoke"``).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

CANONICAL = ("fire", "smoke")


@dataclass(frozen=True)
class Box:
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float

    def yolo_line(self, class_ids: dict[str, int]) -> str:
        cx, cy = (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2
        return f"{class_ids[self.cls]} {cx:.6f} {cy:.6f} {self.x2 - self.x1:.6f} {self.y2 - self.y1:.6f}"


def canonical_class(name: str) -> str | None:
    name = name.strip().lower()
    for canonical in CANONICAL:
        if canonical in name:
            return canonical
    return None


def _clip(v: float) -> float:
    return min(1.0, max(0.0, v))


def read_coco(json_path: Path) -> dict[str, list[Box]]:
    """COCO json: ``bbox = [x, y, w, h]`` in pixels; image size from the ``images`` table."""
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    categories = {c["id"]: canonical_class(c["name"]) for c in payload["categories"]}
    images = {img["id"]: img for img in payload["images"]}
    boxes: dict[str, list[Box]] = defaultdict(list)
    for img in images.values():
        boxes.setdefault(Path(img["file_name"]).stem, [])
    for ann in payload.get("annotations", []):
        cls = categories.get(ann["category_id"])
        if cls is None:
            continue
        img = images[ann["image_id"]]
        w, h = float(img["width"]), float(img["height"])
        x, y, bw, bh = (float(v) for v in ann["bbox"])
        boxes[Path(img["file_name"]).stem].append(
            Box(cls, _clip(x / w), _clip(y / h), _clip((x + bw) / w), _clip((y + bh) / h))
        )
    return dict(boxes)


def read_yolo_dir(labels_dir: Path, class_names: list[str]) -> dict[str, list[Box]]:
    """YOLO txt: one ``class cx cy w h`` line per box, normalised. ``class_names[i]`` names id ``i``."""
    mapping = {i: canonical_class(n) for i, n in enumerate(class_names)}
    boxes: dict[str, list[Box]] = {}
    for txt in Path(labels_dir).rglob("*.txt"):
        if txt.name.lower() in {"classes.txt", "train.txt", "val.txt", "test.txt"}:
            continue
        items = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = mapping.get(int(float(parts[0])))
            if cls is None:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            items.append(Box(cls, _clip(cx - bw / 2), _clip(cy - bh / 2), _clip(cx + bw / 2), _clip(cy + bh / 2)))
        boxes[txt.stem] = items
    return boxes


def read_boxes(path: Path, class_names: list[str] | None = None) -> dict[str, list[Box]]:
    """Dispatch on ``path``: a ``.json`` file is COCO, a directory is YOLO txt labels.

    For YOLO directories the class list is taken from ``class_names`` or a ``classes.txt``
    found in or above the directory; guessing FASDD's id order silently would corrupt labels.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        return read_coco(path)
    if class_names is None:
        for candidate in [path / "classes.txt", path.parent / "classes.txt", path.parent.parent / "classes.txt"]:
            if candidate.exists():
                class_names = [l.strip() for l in candidate.read_text(encoding="utf-8").splitlines() if l.strip()]
                break
    if class_names is None:
        raise ValueError(f"No classes.txt near {path}; pass class_names explicitly (e.g. ['fire', 'smoke']).")
    return read_yolo_dir(path, class_names)


def label_from_boxes(boxes: list[Box]) -> str:
    """Image-level label implied by boxes, using the same rule as ``firecls.data.tdml.infer_label``."""
    classes = {b.cls for b in boxes}
    if {"fire", "smoke"} <= classes:
        return "both"
    if "fire" in classes:
        return "fire"
    if "smoke" in classes:
        return "smoke"
    return "neither"
