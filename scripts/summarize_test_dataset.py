"""Summarise preds/<name>/{cv,rs,uav}.npz dumps (from pipeline/02) on the thesis-comparison test dataset.

    python scripts/summarize_test_dataset.py preds/*_seed42_* --output results/test_dataset_eval
Writes summary.csv (accuracy / macro-F1 / weighted-F1 per model x modality) and summary.md.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

DOMAINS = ["cv", "uav", "rs"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/test_dataset_eval"))
    args = parser.parse_args()
    rows = []
    for d in sorted(args.dirs):
        if not all((d / f"{m}.npz").exists() for m in DOMAINS):
            print(f"skip {d} (incomplete)")
            continue
        for m in DOMAINS:
            z = np.load(d / f"{m}.npz")
            pred, y = z["logits"].argmax(1), z["labels"]
            rows.append({
                "model": d.name, "modality": m, "n": len(y),
                "accuracy": accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro", labels=range(4), zero_division=0),
                "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
            })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| model | modality | n | accuracy | macro-F1 | weighted-F1 |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['modality']} | {r['n']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} |" for r in rows]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
