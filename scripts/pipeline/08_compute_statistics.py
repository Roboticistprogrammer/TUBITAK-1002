"""Classification statistics for the thesis results (issues 2.5, 2.6, 2.7, 2.8). CPU only, numpy only.

Reads the per-image logits written by `02_dump_predictions.py` (any machine, any runtime):

    preds/<group>_seed<N>[_<suffix>]/<domain>.npz        e.g. kd_uniform_seed42, kd_routed_seed43, nokd_seed44,
    preds/teacher_cv|teacher_rs|teacher_uav/<domain>.npz  and TensorRT/ONNX dumps such as kd_uniform_seed42_trt_fp16

and writes, under --out-dir:
  statistics.json          everything (per run, per domain, per group, McNemar tests)
  summary.md               readable tables
  tables/results_summary.tex   rows for Table `tab:results-summary` in Sections/Results.tex

Per run and domain: accuracy, balanced accuracy, per-class precision/recall/F1/support, macro-F1 over the
classes present, confusion matrix, fire-class recall, any-fire recall (fire or both), and the false-alarm rate on
the `neither` class. Per run: equal-weight macro and sample-weighted averages of accuracy (2.7). Per group: mean
and standard deviation over seeds (2.6). Bootstrap 95% CIs for every run's domain, macro and weighted accuracy.
McNemar (exact, two-sided) for student vs in-domain teacher, kd_routed vs kd_uniform and kd_uniform vs nokd,
per seed on identical images. The TensorRT-vs-PyTorch delta (2.8) is the difference between a group and its
`..._trt_fp16` / `..._trt_int8` twin and is printed in summary.md.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

DOMAINS = ["cv", "uav", "rs"]  # column order of the thesis table
CLASSES = ["fire", "smoke", "both", "neither"]
TABLE_ROWS = [  # (group, label) -> rows of tab:results-summary; "teacher_ensemble" is built from the three teachers
    ("teacher_ensemble", "Teachers (domain-matched)"),
    ("kd_uniform", "Student, KD (uniform aggregation)"),
    ("kd_routed", "Student, KD (domain-routed)"),
    ("nokd", "Student, no KD (hard labels only)"),
]


def split_name(name: str) -> tuple[str, int | None]:
    match = re.search(r"_seed(\d+)", name)
    if not match:
        return name, None
    return name[: match.start()] + name[match.end():], int(match.group(1))


def load_run(path: Path) -> dict[str, dict]:
    domains = {}
    for domain in DOMAINS:
        file = path / f"{domain}.npz"
        if file.exists():
            data = np.load(file, allow_pickle=False)
            domains[domain] = {"logits": data["logits"], "labels": data["labels"], "names": data["names"]}
    return domains


def class_metrics(labels: np.ndarray, pred: np.ndarray) -> dict:
    k = len(CLASSES)
    confusion = np.zeros((k, k), dtype=np.int64)
    np.add.at(confusion, (labels, pred), 1)
    support = confusion.sum(1)
    predicted = confusion.sum(0)
    tp = np.diag(confusion).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), np.nan)
    present = support > 0
    fire, both, neither = CLASSES.index("fire"), CLASSES.index("both"), CLASSES.index("neither")
    fire_like = [fire, both]
    any_fire_true = np.isin(labels, fire_like)
    return {
        "samples": int(len(labels)),
        "accuracy": float((labels == pred).mean()),
        "balanced_accuracy": float(np.nanmean(recall[present])),
        "macro_f1_present": float(np.nanmean(np.where(present, np.nan_to_num(f1), np.nan))),
        "per_class": {
            c: {"precision": None if np.isnan(precision[i]) else float(precision[i]),
                "recall": None if np.isnan(recall[i]) else float(recall[i]),
                "f1": None if np.isnan(f1[i]) else float(f1[i]), "support": int(support[i])}
            for i, c in enumerate(CLASSES)
        },
        "fire_recall": None if support[fire] == 0 else float(recall[fire]),
        "any_fire_recall": None if not any_fire_true.any() else float(np.isin(pred[any_fire_true], fire_like).mean()),
        "false_alarm_rate_neither": None if support[neither] == 0 else float(1.0 - recall[neither]),
        "confusion": confusion.tolist(),
    }


def bootstrap_ci(correct: dict[str, np.ndarray], resamples: int, seed: int = 0) -> dict:
    """95% percentile CIs of per-domain, macro and sample-weighted accuracy (resampling images within domain)."""
    rng = np.random.default_rng(seed)
    domains = list(correct)
    sizes = {d: len(correct[d]) for d in domains}
    acc = {d: np.empty(resamples) for d in domains}
    chunk = 250
    for start in range(0, resamples, chunk):
        stop = min(start + chunk, resamples)
        for d in domains:
            idx = rng.integers(0, sizes[d], size=(stop - start, sizes[d]))
            acc[d][start:stop] = correct[d][idx].mean(axis=1)
    macro = np.mean([acc[d] for d in domains], axis=0)
    weighted = sum(acc[d] * sizes[d] for d in domains) / sum(sizes.values())
    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"domains": {d: ci(acc[d]) for d in domains}, "macro": ci(macro), "weighted": ci(weighted), "resamples": resamples}


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    only_a = int(np.sum(a_correct & ~b_correct))
    only_b = int(np.sum(~a_correct & b_correct))
    n = only_a + only_b
    if n == 0:
        return {"only_a_correct": 0, "only_b_correct": 0, "p_value": 1.0}
    k = min(only_a, only_b)
    log_half_n = n * math.log(0.5)
    terms = [math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n) for i in range(k + 1)]
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": float(min(1.0, 2.0 * math.fsum(terms)))}


def evaluate_run(domains: dict[str, dict], resamples: int) -> dict:
    result = {"domains": {}}
    correct = {}
    for d, payload in domains.items():
        pred = payload["logits"].argmax(1)
        result["domains"][d] = class_metrics(payload["labels"], pred)
        correct[d] = payload["labels"] == pred
    accs = [result["domains"][d]["accuracy"] for d in domains]
    sizes = [result["domains"][d]["samples"] for d in domains]
    result["macro_accuracy"] = float(np.mean(accs))
    result["weighted_accuracy"] = float(np.average(accs, weights=sizes))
    result["macro_f1_mean_over_domains"] = float(np.mean([result["domains"][d]["macro_f1_present"] for d in domains]))
    if resamples:
        result["bootstrap_95ci"] = bootstrap_ci(correct, resamples)
    return result, correct


def mean_std(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else None, "n": int(len(arr)), "values": values}


def group_summary(runs: list[dict]) -> dict:
    summary = {"seeds": [r["seed"] for r in runs], "domains": {}}
    for d in DOMAINS:
        if all(d in r["result"]["domains"] for r in runs):
            summary["domains"][d] = {m: mean_std([r["result"]["domains"][d][m] for r in runs])
                                     for m in ("accuracy", "balanced_accuracy", "macro_f1_present")}
    for m in ("macro_accuracy", "weighted_accuracy", "macro_f1_mean_over_domains"):
        summary[m] = mean_std([r["result"][m] for r in runs])
    return summary


def pct(stat: dict | None, latex: bool = True) -> str:
    if stat is None:
        return "--"
    text = f"{100 * stat['mean']:.2f}"
    if stat.get("std") is not None:
        text += (" $\\pm$ " if latex else " ± ") + f"{100 * stat['std']:.2f}"
    return text


def build_teacher_ensemble(teachers: dict[str, dict]) -> dict | None:
    """Domain-matched specialists: teacher_<d> evaluated on domain d."""
    parts = {}
    for d in DOMAINS:
        run = teachers.get(f"teacher_{d}")
        if run is None or d not in run["domains"]:
            return None
        parts[d] = run["domains"][d]
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds-root", type=Path, default=Path("preds"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/statistics"))
    parser.add_argument("--bootstrap", type=int, default=2000, help="resamples per run; 0 disables")
    args = parser.parse_args()

    runs = {}
    for path in sorted(p for p in args.preds_root.iterdir() if p.is_dir() and (p / "meta.json").exists()):
        domains = load_run(path)
        if domains:
            runs[path.name] = domains
    if not runs:
        raise SystemExit(f"no runs under {args.preds_root}")

    reference = {}
    results, correctness, groups = {}, {}, {}
    for name, domains in runs.items():
        for d, payload in domains.items():  # every run must score the same images in the same order
            ref = reference.setdefault(d, payload["names"])
            if len(ref) != len(payload["names"]) or not np.array_equal(ref, payload["names"]):
                raise SystemExit(f"{name}/{d}: image list differs from the other runs; re-dump with the same split/limit")
        results[name], correctness[name] = evaluate_run(domains, args.bootstrap)
        group, seed = split_name(name)
        groups.setdefault(group, []).append({"name": name, "seed": seed, "result": results[name]})

    summaries = {g: group_summary(sorted(rs, key=lambda r: (r["seed"] is None, r["seed"]))) for g, rs in groups.items()}

    teacher_runs = {g: rs[0]["result"] for g, rs in groups.items() if g.startswith("teacher_")}
    ensemble = build_teacher_ensemble(teacher_runs)
    if ensemble:
        accs = [ensemble[d]["accuracy"] for d in DOMAINS]
        sizes = [ensemble[d]["samples"] for d in DOMAINS]
        summaries["teacher_ensemble"] = {
            "seeds": [None],
            "domains": {d: {"accuracy": mean_std([ensemble[d]["accuracy"]]),
                            "macro_f1_present": mean_std([ensemble[d]["macro_f1_present"]])} for d in DOMAINS},
            "macro_accuracy": mean_std([float(np.mean(accs))]),
            "weighted_accuracy": mean_std([float(np.average(accs, weights=sizes))]),
            "macro_f1_mean_over_domains": mean_std([float(np.mean([ensemble[d]["macro_f1_present"] for d in DOMAINS]))]),
        }
        for g, s in summaries.items():
            if g not in teacher_runs and g != "teacher_ensemble" and "trt" not in g and "onnx" not in g:
                s["retention_vs_teacher_ensemble"] = s["macro_accuracy"]["mean"] / summaries["teacher_ensemble"]["macro_accuracy"]["mean"]

    mcnemar = []
    def add_test(label, name_a, name_b, dom_a, dom_b, seed):
        a, b = correctness.get(name_a, {}).get(dom_a), correctness.get(name_b, {}).get(dom_b)
        if a is not None and b is not None and len(a) == len(b):
            mcnemar.append({"comparison": label, "seed": seed, "domain": dom_b, "a": name_a, "b": name_b, **mcnemar_exact(a, b)})
    for g, rs in groups.items():
        if g.startswith("teacher_") or "trt" in g or "onnx" in g:
            continue
        for r in rs:
            for d in DOMAINS:
                add_test(f"{g} vs teacher_{d}", r["name"], f"teacher_{d}", d, d, r["seed"])
    for r in groups.get("kd_routed", []):
        for other in groups.get("kd_uniform", []):
            if other["seed"] == r["seed"]:
                for d in DOMAINS:
                    add_test("kd_routed vs kd_uniform", r["name"], other["name"], d, d, r["seed"])
    for r in groups.get("kd_uniform", []):
        for other in groups.get("nokd", []):
            if other["seed"] == r["seed"]:
                for d in DOMAINS:
                    add_test("kd_uniform vs nokd", r["name"], other["name"], d, d, r["seed"])

    deltas = []
    for g in summaries:
        for kind in ("trt_fp16", "trt_int8", "onnx"):
            twin = next((t for t in summaries if t == f"{g}_{kind}"), None)
            if twin:
                deltas.append({"base": g, "twin": twin, "macro_accuracy_delta_pp": 100 * (summaries[twin]["macro_accuracy"]["mean"] - summaries[g]["macro_accuracy"]["mean"])})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tables").mkdir(exist_ok=True)
    (args.out_dir / "statistics.json").write_text(
        json.dumps({"runs": results, "groups": summaries, "mcnemar": mcnemar, "engine_vs_pytorch": deltas,
                    "classes": CLASSES, "domains": DOMAINS}, indent=2), encoding="utf-8")

    lines = ["# Classification statistics", "", "| group | seeds | " + " | ".join(d.upper() for d in DOMAINS) + " | macro | weighted | macro-F1 |", "|---|---|" + "---|" * (len(DOMAINS) + 3)]
    for g, s in sorted(summaries.items()):
        cells = [pct(s["domains"].get(d, {}).get("accuracy"), latex=False) for d in DOMAINS]
        cells += [pct(s["macro_accuracy"], latex=False), pct(s["weighted_accuracy"], latex=False), pct(s["macro_f1_mean_over_domains"], latex=False)]
        n_seeds = len([x for x in s["seeds"] if x is not None]) or 1
        lines.append(f"| {g} | {n_seeds} | " + " | ".join(cells) + " |")
    if deltas:
        lines += ["", "## Deployed runtime vs PyTorch (macro accuracy delta, percentage points)", ""]
        lines += [f"- {d['twin']} vs {d['base']}: {d['macro_accuracy_delta_pp']:+.3f}" for d in deltas]
    lines += ["", "## McNemar (exact, two-sided)", "", "| comparison | seed | domain | only A correct | only B correct | p |", "|---|---|---|---|---|---|"]
    lines += [f"| {m['comparison']} | {m['seed']} | {m['domain']} | {m['only_a_correct']} | {m['only_b_correct']} | {m['p_value']:.4g} |" for m in mcnemar]
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tex = []
    for group, label in TABLE_ROWS:
        s = summaries.get(group)
        if s is None:
            tex.append(f"{label} & " + " & ".join(["\\tbd"] * 6) + " \\\\")
            continue
        cells = [pct(s["domains"].get(d, {}).get("accuracy")) for d in DOMAINS]
        cells += [pct(s["macro_accuracy"]), pct(s["weighted_accuracy"]), pct(s["macro_f1_mean_over_domains"])]
        tex.append(f"{label} & " + " & ".join(cells) + " \\\\")
    (args.out_dir / "tables" / "results_summary.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print((args.out_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"Wrote {args.out_dir}/statistics.json, summary.md, tables/results_summary.tex")


if __name__ == "__main__":
    main()
