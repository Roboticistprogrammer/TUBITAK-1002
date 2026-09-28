"""Train one Ultralytics detector of the Vazquez et al. (2025) grid on FASDD fire/smoke boxes.

    python scripts/yolo/train_yolo.py --arch yolov8n --init coco --seed 42 --device 0

``--init coco`` fine-tunes the official MS COCO checkpoint (``yolov5nu.pt``, ``yolov8n.pt``,
``yolo11n.pt``; the 80-class head is replaced by a 2-class head); ``--init scratch`` builds the
same architecture from its yaml with random weights. Output: ``<output-root>/<arch>_<init>/seed<k>/``
with Ultralytics' ``weights/best.pt``, ``results.csv``, ``args.yaml`` and our ``train_summary.json``.

Apart from the protocol budget (epochs, batch, imgsz, seed, deterministic) every hyper-parameter
is Ultralytics' default for the installed version and is recorded verbatim in ``args.yaml`` and
``train_summary.json``. Model selection (``best.pt``) is Ultralytics' fitness on the combined
val split; the image-level thresholds are calibrated afterwards on val by
``calibrate_thresholds.py``. A run interrupted mid-way is resumed from ``weights/last.pt``.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import setup_path

ROOT = setup_path()

import torch

from firecls.baselines.families.yolo import ARCHITECTURES, IMG_SIZE
from firecls.baselines.protocol import BATCH_SIZE, EPOCHS
from firecls.deployment.artifacts import sha256_file

SUMMARY_FILE = "train_summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), required=True)
    parser.add_argument("--init", choices=["coco", "scratch"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. 0, 0,1 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="dataloader workers (Ultralytics default 8)")
    parser.add_argument("--optimizer", default="auto",
                        help="Ultralytics default 'auto' (resolution depends on the ultralytics version, see docs)")
    parser.add_argument("--lr0", type=float, default=None, help="only with an explicit --optimizer; default: Ultralytics'")
    parser.add_argument("--momentum", type=float, default=None, help="only with an explicit --optimizer; default: Ultralytics'")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets_yolo" / "fasdd_all.yaml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "baselines" / "yolo")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "third_party" / "weights",
                        help="where COCO checkpoints are downloaded / looked up")
    parser.add_argument("--force", action="store_true", help="retrain even if train_summary.json exists")
    return parser.parse_args(argv)


def run_dir(args: argparse.Namespace) -> Path:
    return (args.output_root / f"{args.arch}_{args.init}" / f"seed{args.seed}").resolve()


def initial_model(args: argparse.Namespace) -> tuple[str, str | None]:
    """Path of the COCO checkpoint (downloaded once into ``--weights-dir``) or the scratch yaml."""
    spec = ARCHITECTURES[args.arch]
    if args.init == "scratch":
        return spec["yaml"], None
    from ultralytics.utils.downloads import attempt_download_asset

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(attempt_download_asset(args.weights_dir / spec["weights"]))
    return str(weights.resolve()), sha256_file(weights)


def is_finished(last: Path) -> bool:
    """Ultralytics marks a completed run by stripping the optimizer and setting ``epoch = -1``."""
    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    return ckpt.get("epoch", -1) == -1


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    import ultralytics
    from ultralytics import YOLO

    out = run_dir(args)
    summary_path = out / SUMMARY_FILE
    best, last = out / "weights" / "best.pt", out / "weights" / "last.pt"
    if summary_path.exists() and best.exists() and not args.force:
        print(f"Skip: {summary_path} exists")
        return best

    model_source, weights_sha = initial_model(args)
    if last.exists() and not args.force and not is_finished(last):
        print(f"Resuming interrupted run from {last}")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        overrides = {k: v for k, v in (("lr0", args.lr0), ("momentum", args.momentum)) if v is not None}
        model = YOLO(model_source)
        model.train(
            **overrides,
            data=str(args.data.resolve()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            optimizer=args.optimizer,
            seed=args.seed,
            deterministic=True,
            pretrained=args.init == "coco",
            project=str(out.parent),  # absolute: Ultralytics re-roots relative projects under its runs_dir
            name=out.name,
            exist_ok=True,
        )
    trainer = model.trainer
    if not best.exists():
        raise RuntimeError(f"Training finished without {best}")

    optimizer = getattr(trainer, "optimizer", None)
    summary = {
        "arch": args.arch,
        "init": args.init,
        "seed": args.seed,
        "initial_model": model_source,
        "initial_weights_sha256": weights_sha,
        "best_checkpoint": str(best),
        "best_checkpoint_sha256": sha256_file(best),
        "best_fitness": float(trainer.best_fitness) if trainer.best_fitness is not None else None,
        "model_selection": "Ultralytics fitness on combined val (fasdd_all.yaml)",
        "resolved_optimizer": type(optimizer).__name__ if optimizer is not None else None,
        "train_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(trainer.args).items()},
        "environment": {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Saved: {best}\nSummary: {summary_path}")
    return best


if __name__ == "__main__":
    main()
