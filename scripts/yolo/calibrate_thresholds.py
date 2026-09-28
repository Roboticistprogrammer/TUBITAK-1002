"""Calibrate the image-level thresholds of a trained YOLO detector on the VALIDATION split.

    python scripts/yolo/calibrate_thresholds.py \
        --checkpoint outputs/baselines/yolo/yolov8n_coco/seed42/weights/best.pt

Runs the detector over each domain's ``val`` images (same CSV index, same letterbox transform
as ``scripts/evaluate_baseline.py``), collects ``(p_fire, p_smoke)`` per image and grid-searches
``(t_fire, t_smoke)`` with :func:`firecls.baselines.scores.calibrate_thresholds`, whose objective
is the equal-weight mean of the metric over domains. The result is written to
``thresholds.json`` beside the checkpoint, keyed by the checkpoint's sha256 so the family
loader refuses thresholds that belong to another checkpoint. The test split is never read.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import setup_path

ROOT = setup_path()

import numpy as np
import torch

from firecls.baselines.data import domain_loaders
from firecls.baselines.families.yolo import IMG_SIZE, THRESHOLDS_FILE, load_presence_model
from firecls.baselines.preprocessing import build_transform, letterbox_spec
from firecls.baselines.protocol import CLASSES, SELECTION_SPLIT
from firecls.baselines.scores import calibrate_thresholds, presence_to_scores
from firecls.deployment.artifacts import sha256_file, write_json
from firecls.evaluation import classification_metrics

METRICS = ("macro_f1_present_classes", "accuracy", "balanced_accuracy")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Ultralytics best.pt")
    parser.add_argument("--dataset-root", type=Path, default=ROOT, help="directory that contains datasets/")
    parser.add_argument("--metric", choices=METRICS, default="macro_f1_present_classes")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None, help=f"default: <checkpoint dir>/{THRESHOLDS_FILE}")
    return parser.parse_args(argv)


@torch.inference_mode()
def collect_presence(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels, fire, smoke = [], [], []
    for images, batch_labels in loader:
        p_fire, p_smoke = model.presence(images.to(device, non_blocking=True))
        fire.append(p_fire.float().cpu().numpy())
        smoke.append(p_smoke.float().cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(labels).astype(np.int64), np.concatenate(fire), np.concatenate(smoke)


def domain_metrics(presence: dict, t_fire: float, t_smoke: float) -> dict:
    out = {}
    for domain, (labels, p_fire, p_smoke) in presence.items():
        predictions = presence_to_scores(p_fire, p_smoke, t_fire, t_smoke).argmax(axis=1)
        out[domain] = classification_metrics(labels, predictions, CLASSES)
    return out


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_presence_model(args.checkpoint)
    model = model.to(device).eval()
    loaders = domain_loaders(
        SELECTION_SPLIT, build_transform(letterbox_spec(IMG_SIZE)), args.batch_size, args.num_workers, args.dataset_root
    )
    presence = {}
    for domain, loader in loaders.items():
        presence[domain] = collect_presence(model, loader, device)
        print(f"{domain}: collected {len(presence[domain][0])} {SELECTION_SPLIT} images")

    best = calibrate_thresholds(presence, metric=args.metric)
    chosen = domain_metrics(presence, best["t_fire"], best["t_smoke"])
    reference = domain_metrics(presence, 0.5, 0.5)
    payload = {
        "t_fire": best["t_fire"],
        "t_smoke": best["t_smoke"],
        "split": SELECTION_SPLIT,
        "objective": {
            "metric": best["metric"],
            "aggregation": "equal-weight mean over domains",
            "value": best["objective"],
            "value_at_0.5_0.5": float(np.mean([m[args.metric] for m in reference.values()])),
        },
        "grid": best["grid"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "val_metrics": chosen,
    }
    output = args.output or args.checkpoint.parent / THRESHOLDS_FILE
    write_json(output, payload)
    for domain, metrics in chosen.items():
        print(f"{domain}: acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1_present_classes']:.4f}")
    print(f"t_fire={best['t_fire']:.2f} t_smoke={best['t_smoke']:.2f} {args.metric}={best['objective']:.4f} -> {output}")
    return output


if __name__ == "__main__":
    main()
