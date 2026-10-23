"""Binary smoke/fire-vs-neither metrics from prediction dumps (external/PLUME_PROTOCOL.md, Sections 1 and 8).

Reads `<preds>/<name>/<domain>.npz` written by `scripts/pipeline/02_dump_predictions.py` (keys logits, labels, names) for the
domain `rs` (FASDD-RS test) or `plume` (Industrial Plumes test groups), and writes metrics.json and summary.md.

    python scripts/evaluate_rs_binary.py --preds preds_before --domain rs \
        --names teacher_rs kd_routed_seed42_trt_fp16 --out-dir results/rs_binary_before

Per model: binary recall, false-alarm rate, balanced accuracy, F1, ROC-AUC with bootstrap 95% CIs (2,000 resamples), error
counts, and for `rs` the recall on the `smoke` and on the `both` images separately. `--pairs a:b` adds the exact McNemar test
of model a against model b on binary correctness, Holm-corrected over all pairs given. Non-significant = inconclusive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bootstrap import setup_path

setup_path()

from firecls import binary_eval as be  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--domain", choices=["rs", "plume"], default="rs")
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--pairs", nargs="*", default=[], help="a:b model pairs for McNemar")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=2000)
    return parser.parse_args()


def load(path: Path, domain: str) -> dict:
    data = np.load(path / f"{domain}.npz", allow_pickle=False)
    if list(data["classes"]) != be.CLASSES:
        raise ValueError(f"{path}: class order differs from {be.CLASSES}")
    logits = data["logits"].astype(np.float64)
    return {"pred": be.positive_prediction(logits), "score": be.positive_score(logits), "labels": data["labels"],
            "truth": be.positive_truth(data["labels"]), "names": data["names"]}


def main() -> None:
    args = parse_args()
    runs = {name: load(args.preds / name, args.domain) for name in args.names}
    first = runs[args.names[0]]
    position = {n: i for i, n in enumerate(first["names"].tolist())}
    for name, run in runs.items():  # dumps from different machines list the images in different orders: align by name
        if sorted(run["names"].tolist()) != sorted(position):
            raise ValueError(f"{name} was not evaluated on the same set of images as {args.names[0]}")
        order = np.argsort([position[n] for n in run["names"].tolist()])
        for key in ("pred", "score", "labels", "truth", "names"):
            run[key] = run[key][order]
        if not np.array_equal(run["labels"], first["labels"]):
            raise ValueError(f"{name}: labels differ from {args.names[0]} for the same image names")
    results = {}
    for name, run in runs.items():
        entry = be.binary_metrics(run["pred"], run["score"], run["truth"])
        entry["ci95"] = be.bootstrap_ci(run["pred"], run["score"], run["truth"], args.resamples)
        entry["images"] = int(len(run["pred"]))
        entry["positives"] = int(run["truth"].sum())
        if args.domain == "rs":
            entry["recall_on_smoke"] = be.recall_on(run["pred"], run["labels"], be.SMOKE)
            entry["recall_on_both"] = be.recall_on(run["pred"], run["labels"], be.BOTH)
        results[name] = entry

    pairs = []
    for spec in args.pairs:
        a, b = spec.split(":")
        pairs.append({"a": a, "b": b, **be.paired_test(runs[a]["pred"], runs[b]["pred"], first["truth"]),
                      "balanced_accuracy_diff_a_minus_b": results[a]["balanced_accuracy"] - results[b]["balanced_accuracy"]})
    for pair, adjusted in zip(pairs, be.holm([p["p_unadjusted"] for p in pairs])):
        pair["p_holm"] = adjusted

    pct = lambda v: f"{100 * v:.1f}"  # noqa: E731
    lines = [f"# Binary smoke/fire vs neither: {args.domain}", "",
             "| Model | Recall % | False-alarm % | Balanced acc. % | F1 % | ROC-AUC | Errors |"
             + (" Recall smoke % | Recall both % |" if args.domain == "rs" else ""),
             "|---|---|---|---|---|---|---|" + ("---|---|" if args.domain == "rs" else "")]
    for name, r in results.items():
        ci = r["ci95"]
        row = (f"| {name} | {pct(r['recall'])} [{pct(ci['recall'][0])}, {pct(ci['recall'][1])}] | "
               f"{pct(r['false_alarm_rate'])} [{pct(ci['false_alarm_rate'][0])}, {pct(ci['false_alarm_rate'][1])}] | "
               f"{pct(r['balanced_accuracy'])} [{pct(ci['balanced_accuracy'][0])}, {pct(ci['balanced_accuracy'][1])}] | "
               f"{pct(r['f1'])} | {r['roc_auc']:.4f} | {r['errors']} / {r['images']} |")
        if args.domain == "rs":
            row += f" {pct(r['recall_on_smoke']['recall'])} ({r['recall_on_smoke']['images']}) | {pct(r['recall_on_both']['recall'])} ({r['recall_on_both']['images']}) |"
        lines.append(row)
    if pairs:
        lines += ["", "Exact McNemar on binary correctness (Holm over the pairs listed):", "",
                  "| A | B | b (A right, B wrong) | c (A wrong, B right) | BA diff (A-B) | p | p Holm |", "|---|---|---|---|---|---|---|"]
        for p in pairs:
            lines.append(f"| {p['a']} | {p['b']} | {p['a_right_b_wrong']} | {p['a_wrong_b_right']} | "
                         f"{100 * p['balanced_accuracy_diff_a_minus_b']:+.1f} | {p['p_unadjusted']:.4f} | {p['p_holm']:.4f} |")
        lines += ["", "A non-significant difference is inconclusive, not evidence of equal performance."]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "metrics.json").write_text(
            json.dumps({"domain": args.domain, "models": results, "mcnemar_holm": pairs}, indent=2) + "\n", encoding="utf-8")
        (args.out_dir / "summary.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
