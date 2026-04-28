from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set

import ijson


@dataclass
class TDMLRecord:
    image_path: Path
    label: str
    split: str


def iter_tdml_items(json_path: Path) -> Iterator[dict]:
    with json_path.open("rb") as f:
        for item in ijson.items(f, "data.item"):
            yield item


def build_filename_index(images_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in images_root.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, path)
    return index


def resolve_image_path(data_url: str, images_root: Path, filename_index: Dict[str, Path]) -> Optional[Path]:
    path = Path(data_url)
    candidate = images_root / path
    if candidate.exists():
        return candidate
    candidate = images_root / path.name
    if candidate.exists():
        return candidate
    return filename_index.get(path.name)


def infer_label(label_items: List[dict], both_as: str) -> str:
    if not label_items:
        return "neither"

    classes: Set[str] = {item.get("class", "").lower() for item in label_items}
    has_fire = "fire" in classes
    has_smoke = "smoke" in classes

    if has_fire and has_smoke:
        return both_as if both_as in {"both", "fire", "smoke"} else "both"
    if has_fire:
        return "fire"
    if has_smoke:
        return "smoke"
    return "neither"


def normalize_split(split: str) -> str:
    split = split.lower()
    if split == "validation":
        return "val"
    if split == "training":
        return "train"
    if split == "test":
        return "test"
    return split


def create_index_csv(
    tdml_json: Path,
    images_root: Path,
    output_csv: Path,
    both_as: str = "both",
    limit: Optional[int] = None,
) -> Dict[str, int]:
    import csv

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filename_index = build_filename_index(images_root)

    counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    missing = 0

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split"])
        writer.writeheader()

        for idx, item in enumerate(iter_tdml_items(tdml_json)):
            if limit is not None and idx >= limit:
                break
            data_url_list = item.get("dataURL", [])
            if not data_url_list:
                continue
            data_url = data_url_list[0]
            resolved = resolve_image_path(data_url, images_root, filename_index)
            if resolved is None:
                missing += 1
                continue

            label = infer_label(item.get("labels", []), both_as)
            split = normalize_split(item.get("trainingType", "train"))
            writer.writerow({"path": str(resolved), "label": label, "split": split})
            counts[split] = counts.get(split, 0) + 1

    counts["missing_images"] = missing
    return counts


def read_tdml_metadata(tdml_json: Path) -> dict:
    with tdml_json.open("r", encoding="utf-8") as f:
        return json.load(f)
