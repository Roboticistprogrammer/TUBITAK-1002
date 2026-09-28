"""Build an Ultralytics detection dataset from the committed image-level splits and FASDD boxes.

The train/val/test membership of every image comes from ``data_index/fasdd_{cv,rs,uav}_tdml.csv``
(the same splits as the teachers, the student and every other baseline); only the *boxes* come
from FASDD's COCO json or YOLO txt annotations. Layout written under ``--output``::

    images/<domain>/<split>/<domain>_<stem><ext>   symlink to the original image (copy fallback)
    labels/<domain>/<split>/<domain>_<stem>.txt    "class cx cy w h" per box, empty for no boxes
    <domain>_<split>.txt                           image lists ("./images/...") read by Ultralytics
    fasdd_all.yaml                                 train = all domains' train, val = all domains' val
    fasdd_<domain>.yaml                            per-domain train/val/test for per-domain metrics
    prepare_report.json                            counts and box-vs-CSV label consistency

Class ids are fixed (fire=0, smoke=1). File names are prefixed with the domain so stems cannot
collide across domains. As a sanity check the image-level label implied by the boxes
(``label_from_boxes``) must agree with the CSV label; the script fails when the mismatch rate
of any domain exceeds ``--max-mismatch-rate``.

Example (box paths are placeholders; point them at your unpacked FASDD COCO json or YOLO dir):

    python scripts/yolo/prepare_fasdd_yolo.py \
        --cv-boxes  <FASDD_CV COCO json> \
        --rs-boxes  <FASDD_RS YOLO labels dir> --yolo-class-names fire smoke \
        --uav-boxes <FASDD_UAV COCO json>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import setup_path

ROOT = setup_path()

import yaml

from firecls.baselines.boxes import Box, label_from_boxes, read_boxes
from firecls.baselines.data import index_csv_path
from firecls.baselines.protocol import DOMAINS
from firecls.config import get_dataset_specs
from firecls.data.dataset import resolve_indexed_image_path

CLASS_IDS = {"fire": 0, "smoke": 1}
SPLITS = ("train", "val", "test")
MAX_EXAMPLES = 20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for domain in DOMAINS:
        parser.add_argument(f"--{domain}-boxes", type=Path, help=f"{domain.upper()} boxes: COCO .json or YOLO labels dir")
    parser.add_argument("--yolo-class-names", nargs="+", default=None,
                        help="class names by YOLO id, for label dirs without classes.txt (e.g. fire smoke)")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--dataset-root", type=Path, default=ROOT, help="directory that contains datasets/")
    parser.add_argument("--index-root", type=Path, default=ROOT, help="directory that contains data_index/")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets_yolo")
    parser.add_argument("--copy", action="store_true", help="copy images instead of symlinking them")
    parser.add_argument("--max-mismatch-rate", type=float, default=0.01)
    return parser.parse_args(argv)


def read_index(csv_path: Path) -> list[dict]:
    with Path(csv_path).open("r", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["split"].lower() in SPLITS]


def place_image(source: Path, target: Path, copy: bool) -> str:
    """Symlink ``target -> source``; copy when symlinks are unavailable (e.g. Windows without privileges)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    if not copy:
        try:
            os.symlink(source.resolve(), target)
            return "symlink"
        except OSError:
            pass
    shutil.copy2(source, target)
    return "copy"


def valid_boxes(boxes: list[Box]) -> tuple[list[Box], int]:
    """Drop boxes that collapse to zero area after clipping; they carry no signal for the detector."""
    kept = [b for b in boxes if b.x2 > b.x1 and b.y2 > b.y1 and b.cls in CLASS_IDS]
    return kept, len(boxes) - len(kept)


def prepare_domain(
    domain: str, boxes_by_stem: dict[str, list[Box]], images_root: Path, csv_path: Path, output: Path, copy: bool
) -> dict:
    rows = read_index(csv_path)
    stems = Counter(PureWindowsPath(row["path"]).stem for row in rows)
    duplicates = sorted(stem for stem, count in stems.items() if count > 1)
    if duplicates:
        raise ValueError(f"{csv_path}: {len(duplicates)} duplicate image stems (e.g. {duplicates[:3]}); boxes are keyed by stem")

    lists: dict[str, list[str]] = {split: [] for split in SPLITS}
    report = {
        "csv": str(csv_path),
        "images": Counter(),
        "boxes": Counter(),
        "images_without_annotation_entry": 0,
        "degenerate_boxes_dropped": 0,
        "mismatches": 0,
        "mismatch_confusion": Counter(),
        "mismatch_examples": [],
        "missing_images": [],
        "placement": Counter(),
    }
    for row in rows:
        split, csv_label = row["split"].lower(), row["label"].lower()
        try:
            source = resolve_indexed_image_path(row["path"], images_root)
        except FileNotFoundError:
            report["missing_images"].append(row["path"])
            continue
        stem = PureWindowsPath(row["path"]).stem
        if stem not in boxes_by_stem:
            report["images_without_annotation_entry"] += 1
        boxes, dropped = valid_boxes(boxes_by_stem.get(stem, []))
        report["degenerate_boxes_dropped"] += dropped

        name = f"{domain}_{stem}"
        image_target = output / "images" / domain / split / f"{name}{source.suffix.lower()}"
        label_target = output / "labels" / domain / split / f"{name}.txt"
        report["placement"][place_image(source, image_target, copy)] += 1
        label_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.write_text("".join(b.yolo_line(CLASS_IDS) + "\n" for b in boxes), encoding="utf-8")
        lists[split].append("./" + image_target.relative_to(output).as_posix())

        report["images"][split] += 1
        report["boxes"].update(b.cls for b in boxes)
        box_label = label_from_boxes(boxes)
        if box_label != csv_label:
            report["mismatches"] += 1
            report["mismatch_confusion"][f"csv={csv_label} boxes={box_label}"] += 1
            if len(report["mismatch_examples"]) < MAX_EXAMPLES:
                report["mismatch_examples"].append({"path": row["path"], "csv": csv_label, "boxes": box_label})

    if report["missing_images"]:
        raise FileNotFoundError(
            f"{domain}: {len(report['missing_images'])} indexed images not found beneath {images_root} "
            f"(e.g. {report['missing_images'][:3]})"
        )
    for split, entries in lists.items():
        (output / f"{domain}_{split}.txt").write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")
    total = sum(report["images"].values())
    report["mismatch_rate"] = report["mismatches"] / total if total else 0.0
    return {k: dict(v) if isinstance(v, Counter) else v for k, v in report.items()}


def write_yaml(path: Path, output: Path, splits: dict[str, list[str]]) -> None:
    payload = {"path": str(output.resolve()), **splits, "names": {i: n for n, i in CLASS_IDS.items()}}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    specs = get_dataset_specs(args.dataset_root)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    for cache in (output / "labels").rglob("*.cache") if (output / "labels").exists() else []:
        cache.unlink()  # Ultralytics' label cache hashes file sizes only; stale caches would survive relabelling

    domains: dict[str, dict] = {}
    for domain in args.domains:
        boxes_path = getattr(args, f"{domain}_boxes")
        if boxes_path is None:
            raise SystemExit(f"--{domain}-boxes is required (or restrict --domains)")
        boxes = read_boxes(boxes_path, args.yolo_class_names)
        report = prepare_domain(
            domain, boxes, specs[domain].images_root, index_csv_path(domain, args.index_root), output, args.copy
        )
        report["boxes_source"] = str(boxes_path)
        domains[domain] = report
        print(f"{domain}: images={report['images']} boxes={report['boxes']} "
              f"mismatches={report['mismatches']} ({report['mismatch_rate']:.4%}) "
              f"no_annotation_entry={report['images_without_annotation_entry']} "
              f"degenerate_dropped={report['degenerate_boxes_dropped']}")
        write_yaml(output / f"fasdd_{domain}.yaml", output, {s: f"{domain}_{s}.txt" for s in SPLITS})

    write_yaml(
        output / "fasdd_all.yaml",
        output,
        {"train": [f"{d}_train.txt" for d in args.domains], "val": [f"{d}_val.txt" for d in args.domains]},
    )
    summary = {
        "class_ids": CLASS_IDS,
        "max_mismatch_rate": args.max_mismatch_rate,
        "dataset_root": str(args.dataset_root.resolve()),
        "domains": domains,
    }
    (output / "prepare_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    too_high = {d: r["mismatch_rate"] for d, r in domains.items() if r["mismatch_rate"] > args.max_mismatch_rate}
    if too_high:
        raise SystemExit(
            f"Box-implied labels disagree with the CSV labels above {args.max_mismatch_rate:.2%}: {too_high}. "
            f"See {output / 'prepare_report.json'} (wrong boxes file or class-name order?)."
        )
    print(f"Wrote {output / 'fasdd_all.yaml'} and per-domain yamls")
    return summary


if __name__ == "__main__":
    main()
