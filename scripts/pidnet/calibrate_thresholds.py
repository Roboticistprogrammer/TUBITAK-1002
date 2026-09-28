"""Calibrate the PIDNet presence thresholds (t_fire, t_smoke) on the VALIDATION split.

Presence probabilities are computed with the exact evaluation transform (letterbox, via
``firecls.baselines.data.domain_loaders``), the thresholds are grid-searched with
``firecls.baselines.scores.calibrate_thresholds`` (equal-weight domain mean of the objective),
and ``thresholds.json`` is written beside the checkpoint, where the ``pidnet`` family loader
reads it. The test split is never used; there is deliberately no ``--split`` option.

    python scripts/pidnet/calibrate_thresholds.py --checkpoint outputs/baselines/pidnet_s/seed42/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import setup_path  # noqa: E402

ROOT = setup_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from firecls.baselines.data import domain_loaders  # noqa: E402
from firecls.baselines.pidnet.checkpoint import load_checkpoint, scorer_from_checkpoint, write_thresholds  # noqa: E402
from firecls.baselines.preprocessing import build_transform, letterbox_spec  # noqa: E402
from firecls.baselines.protocol import CLASSES, SELECTION_SPLIT  # noqa: E402
from firecls.baselines.scores import calibrate_thresholds, presence_to_scores  # noqa: E402
from firecls.evaluation import classification_metrics  # noqa: E402

METRICS = ["macro_f1_present_classes", "accuracy", "balanced_accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT, help="directory that contains datasets/")
    parser.add_argument("--metric", choices=METRICS, default=METRICS[0], help="calibration objective")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def summarise(domain_presence: dict, t_fire: float, t_smoke: float) -> dict:
    out = {}
    for domain, (labels, p_fire, p_smoke) in domain_presence.items():
        metrics = classification_metrics(labels, presence_to_scores(p_fire, p_smoke, t_fire, t_smoke).argmax(1), CLASSES)
        out[domain] = {k: metrics[k] for k in ("samples", "accuracy", "balanced_accuracy", "macro_f1_present_classes")}
    return out


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = load_checkpoint(args.checkpoint)
    model = scorer_from_checkpoint(payload).to(device).eval()  # thresholds irrelevant for presence
    spec = letterbox_spec(int(payload["img_size"]), int(payload.get("pad_value", 114)))
    loaders = domain_loaders(SELECTION_SPLIT, build_transform(spec), args.batch_size, args.num_workers, args.dataset_root)

    domain_presence = {}
    for domain, loader in loaders.items():
        labels, p_fire, p_smoke = [], [], []
        for images, batch_labels in loader:
            fire, smoke = model.presence(images.to(device, non_blocking=True))
            p_fire.append(fire.float().cpu().numpy())
            p_smoke.append(smoke.float().cpu().numpy())
            labels.append(batch_labels.numpy())
        domain_presence[domain] = (np.concatenate(labels), np.concatenate(p_fire), np.concatenate(p_smoke))
        print(f"{domain}: {domain_presence[domain][0].size} val images")

    best = calibrate_thresholds(domain_presence, metric=args.metric)
    np.savez_compressed(
        args.checkpoint.parent / "val_presence.npz",
        **{f"{d}_{name}": arr for d, arrays in domain_presence.items() for name, arr in zip(("labels", "p_fire", "p_smoke"), arrays)},
    )
    path = write_thresholds(
        args.checkpoint,
        {
            **best,
            "split": SELECTION_SPLIT,
            "img_size": int(payload["img_size"]),
            "topk_fraction": float(payload["topk_fraction"]),
            "val_at_calibrated": summarise(domain_presence, best["t_fire"], best["t_smoke"]),
            "val_at_default_0.5": summarise(domain_presence, 0.5, 0.5),
        },
    )
    print(f"t_fire={best['t_fire']:.2f} t_smoke={best['t_smoke']:.2f} {args.metric}={best['objective']:.4f} -> {path}")


if __name__ == "__main__":
    main()
