"""Export any registered baseline family to a verified ONNX file with a manifest.

    python scripts/export_baseline.py --family cnn \
        --checkpoint outputs/baselines/resnet50/seed42/best.pt \
        --output artifacts/baselines/resnet50_seed42.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bootstrap import setup_path

ROOT = setup_path()

from firecls.baselines.export import export_scores_model
from firecls.baselines.protocol import SOURCES
from firecls.baselines.registry import available_families, load_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", required=True, help=f"one of {available_families()}")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=SOURCES, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fixed-batch", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_baseline(args.family, args.checkpoint, "cpu")
    export_model = getattr(loaded.model, "export_module", lambda: loaded.model)()
    path = export_scores_model(
        export_model,
        args.output,
        loaded.preprocessing,
        args.checkpoint,
        family=args.family,
        model_name=loaded.metadata.get("model_name", args.family),
        source=args.source or loaded.metadata.get("source", "official"),
        opset=args.opset,
        dynamic_batch=not args.fixed_batch,
        atol=args.atol,
        extra={k: v for k, v in loaded.metadata.items() if k in ("thresholds", "train_seed", "epoch")},
    )
    print(f"Exported {args.family} baseline: {path}")


if __name__ == "__main__":
    main()
