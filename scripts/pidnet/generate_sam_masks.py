"""Generate SAM (ViT-H) box-prompted pseudo-masks for the train and val splits of all domains.

Each ground-truth box is one SAM prompt (``multimask_output=False``); per-class unions are stored
as ``pseudo_masks/<domain>/<stem>.png`` with bit0 = fire, bit1 = smoke. The test split is never
touched. The run is resumable: existing PNGs are skipped, so the command can simply be repeated.

    python scripts/pidnet/generate_sam_masks.py \
        --sam-checkpoint third_party/weights/sam_vit_h_4b8939.pth \
        --cv-boxes <FASDD_CV COCO json(s) or YOLO labels dir> \
        --rs-boxes <FASDD_RS ...> --uav-boxes <FASDD_UAV ...> \
        --yolo-class-names fire smoke   # only for YOLO dirs without classes.txt
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import setup_path  # noqa: E402

ROOT = setup_path()

import json  # noqa: E402

from tqdm import tqdm  # noqa: E402

from firecls.baselines.boxes import Box, read_boxes  # noqa: E402
from firecls.baselines.data import build_domain_datasets  # noqa: E402
from firecls.baselines.pidnet.masks import SamBoxSegmenter, generate_masks, mask_path  # noqa: E402
from firecls.baselines.protocol import CLASSES, DOMAINS  # noqa: E402
from firecls.deployment.artifacts import git_commit, sha256_file, write_json  # noqa: E402

TRAINING_SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sam-checkpoint", type=Path, required=True, help="sam_vit_h_4b8939.pth")
    parser.add_argument("--sam-model", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-root", type=Path, default=ROOT, help="directory that contains datasets/")
    parser.add_argument("--masks-root", type=Path, default=ROOT / "pseudo_masks")
    for domain in DOMAINS:
        parser.add_argument(
            f"--{domain}-boxes", type=Path, nargs="+", default=None,
            help="COCO .json file(s) or YOLO labels directory(ies); several are merged (e.g. per-split files)",
        )
    parser.add_argument("--yolo-class-names", nargs="+", default=None, help="YOLO id order, e.g. fire smoke")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=DOMAINS)
    parser.add_argument("--splits", nargs="+", choices=TRAINING_SPLITS, default=list(TRAINING_SPLITS))
    parser.add_argument("--limit", type=int, default=None, help="max masks to write in this run (for a dry run)")
    return parser.parse_args()


def load_domain_boxes(paths: list[Path], class_names: list[str] | None) -> dict[str, list[Box]]:
    merged: dict[str, list[Box]] = {}
    for path in paths:
        for stem, boxes in read_boxes(path, class_names).items():
            merged.setdefault(stem, []).extend(boxes)
    return merged


def main() -> None:
    args = parse_args()
    boxes, samples, by_split = {}, {}, {}
    for domain in args.domains:
        paths = getattr(args, f"{domain}_boxes")
        if not paths:
            raise SystemExit(f"--{domain}-boxes is required for domain '{domain}'")
        boxes[domain] = load_domain_boxes(paths, args.yolo_class_names)
        samples[domain] = []
        for split in args.splits:
            dataset = build_domain_datasets(split, None, args.dataset_root, domains=[domain])[domain]
            by_split[f"{domain}/{split}"] = [path for path, _ in dataset.samples]
            samples[domain].extend((path, CLASSES[label]) for path, label in dataset.samples)
        print(f"{domain}: {len(samples[domain])} images in {args.splits}, box entries for {len(boxes[domain])} stems")

    pending = sum(not mask_path(args.masks_root, d, p).exists() for d, items in samples.items() for p, _ in items)
    print(f"{pending} masks to generate under {args.masks_root}")
    started = time.perf_counter()
    segmenter = SamBoxSegmenter(args.sam_checkpoint, args.sam_model, args.device) if pending else None
    load_seconds = time.perf_counter() - started
    with tqdm(total=sum(len(v) for v in samples.values()), unit="img") as progress:
        stats = generate_masks(samples, boxes, segmenter, args.masks_root, args.limit, progress) if pending else {}
    elapsed = time.perf_counter() - started

    expected = {key: len(paths) for key, paths in by_split.items()}
    present = {
        key: sum(mask_path(args.masks_root, key.split("/")[0], p).exists() for p in paths)
        for key, paths in by_split.items()
    }
    manifest_path = args.masks_root / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    run = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(ROOT),
        "sam_model": args.sam_model,
        "sam_checkpoint": str(args.sam_checkpoint),
        "sam_checkpoint_sha256": sha256_file(args.sam_checkpoint),
        "device": args.device,
        "splits": args.splits,
        "box_sources": {d: [str(p) for p in getattr(args, f"{d}_boxes")] for d in args.domains},
        "limit": args.limit,
        "model_load_seconds": load_seconds,
        "total_seconds": elapsed,
        "per_domain": stats,
    }
    manifest = {
        "encoding": {"format": "uint8 PNG, original image resolution", "bit0": "fire", "bit1": "smoke"},
        "prompting": "one box prompt per ground-truth box, multimask_output=False; per-class union; "
        "filled box if SAM fails or returns an empty mask",
        "expected": {**previous.get("expected", {}), **expected},
        "present": {**previous.get("present", {}), **present},
        "runs": previous.get("runs", []) + [run],
    }
    required = {f"{d}/{s}" for d in DOMAINS for s in TRAINING_SPLITS}
    manifest["complete"] = required <= set(manifest["expected"]) and all(
        manifest["present"].get(key, 0) >= manifest["expected"][key] for key in required
    )
    write_json(manifest_path, manifest)
    for domain, s in stats.items():
        per_image = s["seconds"] / max(1, s["written"])
        print(
            f"{domain}: written={s['written']} skipped={s['skipped_existing']} boxes={s['boxes']} "
            f"fallback(error/empty)={s['fallback_error']}/{s['fallback_empty']} label_mismatch={s['label_mismatch']} "
            f"missing_box_entry={s['missing_box_entry']} ({per_image:.2f} s/img)"
        )
    print(f"Manifest: {manifest_path} (complete={manifest['complete']})")


if __name__ == "__main__":
    main()
