"""Metrics and statistics for the USTC_SmokeRS external validation (rules: external/USTC_PROTOCOL.md, sections 4-7).

Reads the dumps written by `scripts/pipeline/02_dump_predictions.py --dataset ustc` and writes, under --out-dir:
  metrics.json   everything: per cell, per model, per negative class, bootstrap CIs, McNemar pairs
  summary.md     readable tables, one section per cell
  false_alarm_by_class_<cell>.png   false-alarm rate per negative class and model (if matplotlib is installed)
  top_errors.json   highest-confidence misses and false alarms per model, main cell (to pick the 4-6 example images)

A "cell" is one (preprocessing, prediction rule) combination. All of them are computed; only (primary, primary) is the
main result, the others are sensitivity analyses:
  preprocessing primary      --preds              resize 224 + centre crop 192 (the FASDD transform)
  preprocessing full_resize  --preds-full-resize  whole image resized to 192 x 192 (optional; dumped with
                                                  02_dump_predictions.py --preprocessing full_resize)
  rule primary    positive = fire, smoke or both     score for ROC-AUC = 1 - P(neither)
  rule secondary  positive = smoke or both           score for ROC-AUC = P(smoke) + P(both)

    python scripts/evaluate_external.py --preds preds_ustc --preds-full-resize preds_ustc_full_resize \
        --names kd_routed_seed42 kd_uniform_seed42 nokd_seed42 mobilenet_v3_large_seed42 resnet50_seed42 \
        --reference teacher_rs --out-dir results/external/ustc_smokers

`--reference` models (the RS teacher) are reported but left out of the pairwise tests, which cover `--names` only; the
Holm correction is applied within each cell. No threshold or calibration is tuned here.
"""
from __future__ import annotations

import argparse
import itertools
import json
from math import comb
from pathlib import Path

import numpy as np

CLASSES = ["fire", "smoke", "both", "neither"]
FIRE, SMOKE, BOTH, NEITHER = (CLASSES.index(c) for c in ("fire", "smoke", "both", "neither"))
USTC_CLASSES = ["smoke", "cloud", "dust", "haze", "land", "seaside"]
NEGATIVES = USTC_CLASSES[1:]
RULES = ("primary", "secondary")
MAIN_CELL = "primary_preprocessing/primary_rule"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds", type=Path, default=Path("preds"), help="dumps with the primary preprocessing")
    parser.add_argument("--preds-full-resize", type=Path, default=None, help="dumps with the secondary preprocessing")
    parser.add_argument("--names", nargs="+", required=True, help="models in the pairwise tests (the five deployed ones)")
    parser.add_argument("--reference", nargs="*", default=[], help="reported only, e.g. teacher_rs")
    parser.add_argument("--out-dir", type=Path, default=Path("results/external/ustc_smokers"))
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load(path: Path) -> dict:
    data = np.load(path / "ustc.npz", allow_pickle=False)
    if list(data["classes"]) != CLASSES or list(data["classes6"]) != USTC_CLASSES:
        raise ValueError(f"{path}: class order differs from the protocol")
    logits = data["logits"].astype(np.float64)
    return {"pred": logits.argmax(1), "prob": softmax(logits), "labels6": data["labels6"],
            "labels_bin": data["labels_bin"].astype(bool), "names": data["names"]}


def apply_rule(run: dict, rule: str) -> dict:
    """Binary prediction and threshold-free score of one prediction rule (protocol, Section 5)."""
    pred, prob = run["pred"], run["prob"]
    if rule == "primary":
        positive, score = pred != NEITHER, 1.0 - prob[:, NEITHER]
    elif rule == "secondary":
        positive, score = np.isin(pred, [SMOKE, BOTH]), prob[:, SMOKE] + prob[:, BOTH]
    else:
        raise ValueError(rule)
    return {**run, "pos_pred": positive, "score": score}


def roc_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUC with average ranks for ties."""
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def core_metrics(pos_pred: np.ndarray, score: np.ndarray, labels_bin: np.ndarray) -> dict:
    recall = float(pos_pred[labels_bin].mean())
    far = float(pos_pred[~labels_bin].mean())
    return {"smoke_recall": recall, "false_alarm_rate": far, "balanced_accuracy": (recall + 1 - far) / 2,
            "roc_auc": roc_auc(score, labels_bin)}


def bootstrap_ci(run: dict, resamples: int, rng: np.random.Generator) -> dict:
    n = len(run["pos_pred"])
    samples = {key: [] for key in ("smoke_recall", "false_alarm_rate", "balanced_accuracy", "roc_auc")}
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        metrics = core_metrics(run["pos_pred"][idx], run["score"][idx], run["labels_bin"][idx])
        for key, value in metrics.items():
            samples[key].append(value)
    return {key: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for key, v in samples.items()}


def evaluate(run: dict, resamples: int, rng: np.random.Generator) -> dict:
    labels6, pos_pred, pred = run["labels6"], run["pos_pred"], run["pred"]
    smoke = labels6 == USTC_CLASSES.index("smoke")
    per_class = {}
    for name in NEGATIVES:
        mask = labels6 == USTC_CLASSES.index(name)
        per_class[name] = {"images": int(mask.sum()),
                           "false_alarm_rate": float(pos_pred[mask].mean()) if mask.any() else float("nan")}
    distribution = {c: float((pred[smoke] == i).mean()) for i, c in enumerate(CLASSES)} if smoke.any() else {}
    result = core_metrics(pos_pred, run["score"], run["labels_bin"])
    result.update({
        "images": int(len(pred)), "smoke_images": int(smoke.sum()), "negative_images": int((~smoke).sum()),
        "false_alarm_by_class": per_class, "predicted_class_share_on_smoke": distribution,
        "predicted_class_share_overall": {c: float((pred == i).mean()) for i, c in enumerate(CLASSES)},
        "ci95": bootstrap_ci(run, resamples, rng) if resamples else None,
    })
    return result


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail / 2 ** n)


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def pairwise(runs: dict[str, dict], names: list[str]) -> list[dict]:
    rows = []
    for a, b in itertools.combinations(names, 2):
        if not np.array_equal(runs[a]["names"], runs[b]["names"]):
            raise ValueError(f"{a} and {b} were not evaluated on identical images in identical order")
        truth = runs[a]["labels_bin"]
        ok_a, ok_b = runs[a]["pos_pred"] == truth, runs[b]["pos_pred"] == truth
        only_a, only_b = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
        ba = lambda r: core_metrics(r["pos_pred"], r["score"], r["labels_bin"])["balanced_accuracy"]  # noqa: E731
        rows.append({"a": a, "b": b, "a_correct_b_wrong": only_a, "a_wrong_b_correct": only_b,
                     "balanced_accuracy_diff_a_minus_b": ba(runs[a]) - ba(runs[b]),
                     "p_unadjusted": mcnemar_exact(only_a, only_b)})
    for row, adjusted in zip(rows, holm([r["p_unadjusted"] for r in rows])):
        row["p_holm"] = adjusted
    return rows


def top_errors(run: dict, k: int = 6) -> dict:
    wrong = {}
    miss = np.where(run["labels_bin"] & ~run["pos_pred"])[0]
    false = np.where(~run["labels_bin"] & run["pos_pred"])[0]
    for key, idx, sign in (("missed_smoke", miss, 1), ("false_alarm", false, -1)):
        order = idx[np.argsort(sign * run["score"][idx])][:k]  # most confident mistakes first
        wrong[key] = [{"image": str(run["names"][i]), "score": float(run["score"][i]),
                       "predicted": CLASSES[int(run["pred"][i])]} for i in order]
    return wrong


def fmt(value: float, ci: list[float] | None = None) -> str:
    return f"{value:.3f}" if ci is None else f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def summary_section(cell: str, results: dict, pairs: list[dict], names: list[str]) -> list[str]:
    main = " (MAIN RESULT)" if cell == MAIN_CELL else " (sensitivity analysis)"
    lines = [f"## {cell}{main}", "", "| Model | Smoke recall | False-alarm rate | Balanced acc. | ROC-AUC |", "|---|---|---|---|---|"]
    for name, r in results.items():
        ci = r["ci95"] or {}
        tag = "" if name in names else " (reference)"
        lines.append(f"| {name}{tag} | {fmt(r['smoke_recall'], ci.get('smoke_recall'))} | "
                     f"{fmt(r['false_alarm_rate'], ci.get('false_alarm_rate'))} | "
                     f"{fmt(r['balanced_accuracy'], ci.get('balanced_accuracy'))} | "
                     f"{fmt(r['roc_auc'], ci.get('roc_auc'))} |")
    lines += ["", "False-alarm rate per negative class:", "",
              "| Model | " + " | ".join(NEGATIVES) + " |", "|---|" + "---|" * len(NEGATIVES)]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r['false_alarm_by_class'][c]['false_alarm_rate']:.3f}" for c in NEGATIVES) + " |")
    lines += ["", "Predicted class on smoke images (share):", "", "| Model | " + " | ".join(CLASSES) + " |",
              "|---|" + "---|" * len(CLASSES)]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r['predicted_class_share_on_smoke'].get(c, float('nan')):.3f}" for c in CLASSES) + " |")
    if pairs:
        lines += ["", "Pairwise exact McNemar (binary correctness), Holm-adjusted within this cell:", "",
                  "| A | B | b (A right, B wrong) | c (A wrong, B right) | BA diff (A-B) | p | p Holm |", "|---|---|---|---|---|---|---|"]
        for p in pairs:
            lines.append(f"| {p['a']} | {p['b']} | {p['a_correct_b_wrong']} | {p['a_wrong_b_correct']} | "
                         f"{p['balanced_accuracy_diff_a_minus_b']:+.3f} | {p['p_unadjusted']:.4f} | {p['p_holm']:.4f} |")
    return lines + [""]


def plot_false_alarms(path: Path, results: dict) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(1, len(results))
    for i, (name, r) in enumerate(results.items()):
        values = [r["false_alarm_by_class"][c]["false_alarm_rate"] for c in NEGATIVES]
        ax.bar(np.arange(len(NEGATIVES)) + i * width, values, width, label=name)
    ax.set_xticks(np.arange(len(NEGATIVES)) + 0.4 - width / 2, NEGATIVES)
    ax.set_ylabel("False-alarm rate")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    everything = list(dict.fromkeys(args.names + args.reference))
    sources = {"primary_preprocessing": args.preds}
    if args.preds_full_resize is not None:
        sources["full_resize_preprocessing"] = args.preds_full_resize
    rng = np.random.default_rng(args.seed)
    cells, summary, plotted = {}, ["# USTC_SmokeRS external validation", "",
                                   "Positive = predicted fire/smoke/both (primary rule) or smoke/both (secondary rule). "
                                   "Brackets: 95% bootstrap CI over images.", ""], True
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for preprocessing, root in sources.items():
        raw = {name: load(root / name) for name in everything}
        for rule in RULES:
            cell = f"{preprocessing}/{rule}_rule"
            runs = {name: apply_rule(run, rule) for name, run in raw.items()}
            results = {name: evaluate(run, args.resamples, rng) for name, run in runs.items()}
            pairs = pairwise(runs, args.names) if len(args.names) > 1 else []
            cells[cell] = {"preprocessing": preprocessing, "rule": rule, "models": results, "mcnemar_holm": pairs}
            summary += summary_section(cell, results, pairs, args.names)
            plotted &= plot_false_alarms(args.out_dir / f"false_alarm_by_class_{cell.replace('/', '__')}.png", results)
            if cell == MAIN_CELL:
                (args.out_dir / "top_errors.json").write_text(
                    json.dumps({n: top_errors(r) for n, r in runs.items()}, indent=2) + "\n", encoding="utf-8")
    summary.append("A non-significant difference is inconclusive, not evidence of equal performance.")
    (args.out_dir / "metrics.json").write_text(
        json.dumps({"protocol": "external/USTC_PROTOCOL.md", "main_cell": MAIN_CELL, "resamples": args.resamples,
                    "seed": args.seed, "cells": cells}, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"Wrote {args.out_dir}" + ("" if plotted else " (matplotlib missing: no figure)"))


if __name__ == "__main__":
    main()
