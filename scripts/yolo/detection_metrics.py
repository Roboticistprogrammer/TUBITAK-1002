"""Box-level detector quality on each domain's TEST split (secondary table, not the main comparison).

    python scripts/yolo/detection_metrics.py --name yolov8n_coco_seed42 \
        --checkpoint outputs/baselines/yolo/yolov8n_coco/seed42/weights/best.pt \
        --output results/baselines/yolov8n_coco_seed42_detection.json

Runs Ultralytics ``val`` with its default settings (conf 0.001, NMS IoU 0.7, max_det 300,
rectangular batches) on ``datasets_yolo/fasdd_<domain>.yaml`` with ``split=test`` and records
precision, recall, mAP50 and mAP50-95, overall and per class. These are the metrics Vazquez et
al. report; the image-level metrics of the main table come from ``scripts/evaluate_baseline.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import setup_path

ROOT = setup_path()

import numpy as np

from firecls.baselines.families.yolo import IMG_SIZE
from firecls.baselines.protocol import DOMAINS, EVAL_SPLIT
from firecls.deployment.artifacts import git_commit, sha256_file, write_json

METRIC_NAMES = ("precision", "recall", "mAP50", "mAP50-95")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="row name, e.g. yolov8n_coco_seed42")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets_yolo", help="output of prepare_fasdd_yolo.py")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--split", default=EVAL_SPLIT)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs" / "baselines" / "yolo" / "detection_val",
                        help="scratch directory for Ultralytics' val outputs")
    return parser.parse_args(argv)


def summarise(metrics) -> dict:
    box = metrics.box
    names = metrics.names
    per_class = {
        names[int(c)]: dict(zip(METRIC_NAMES, (float(v) for v in box.class_result(i))))
        for i, c in enumerate(box.ap_class_index)
    }
    instances = metrics.nt_per_class if metrics.nt_per_class is not None else np.zeros(len(names))
    images = metrics.nt_per_image if metrics.nt_per_image is not None else np.zeros(len(names))
    for index, name in names.items():
        per_class.setdefault(name, {k: None for k in METRIC_NAMES})  # class absent from this split (e.g. RS fire)
        per_class[name]["instances"] = int(instances[index])
        per_class[name]["images_with_class"] = int(images[index])
    return {
        **dict(zip(METRIC_NAMES, (float(box.mp), float(box.mr), float(box.map50), float(box.map)))),
        "per_class": per_class,
        "speed_ms_per_image": {k: float(v) for k, v in metrics.speed.items()},
    }


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    import ultralytics
    from ultralytics import YOLO

    run_root = (args.runs_dir / args.name).resolve()
    payload = {
        "name": args.name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "split": args.split,
        "settings": {"imgsz": args.imgsz, "batch": args.batch, "conf": 0.001, "iou": 0.7, "max_det": 300},
        "ultralytics": ultralytics.__version__,
        "git_commit": git_commit(ROOT),
        "domains": {},
    }
    for domain in args.domains:
        data = (args.data_dir / f"fasdd_{domain}.yaml").resolve()
        metrics = YOLO(str(args.checkpoint)).val(
            data=str(data),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(run_root),  # absolute: Ultralytics re-roots relative projects under its runs_dir
            name=domain,
            exist_ok=True,
            plots=False,
            verbose=False,
        )
        payload["domains"][domain] = summarise(metrics)
        d = payload["domains"][domain]
        print(f"{args.name}/{domain}: P={d['precision']:.4f} R={d['recall']:.4f} "
              f"mAP50={d['mAP50']:.4f} mAP50-95={d['mAP50-95']:.4f}")
    domains = payload["domains"].values()
    payload["domain_macro"] = {n: float(np.mean([d[n] for d in domains])) for n in METRIC_NAMES}
    write_json(args.output, payload)
    print(f"Saved: {args.output}")
    return payload


if __name__ == "__main__":
    main()
