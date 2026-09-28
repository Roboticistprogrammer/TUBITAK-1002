"""Merge evaluation and benchmark JSONs into one thesis table (LaTeX booktabs + CSV).

Rows are grouped by name with any ``_seedNN`` suffix removed, so three seeds of one method
become a single ``mean +- std`` row. The provenance marker of each row is appended to the
method name: none for official code, a dagger for reimplementations, a double dagger for
numbers copied from a paper.

Example:
    python scripts/collect_results_table.py \
        --eval results/baselines/*_eval.json results/classification_evaluation.json \
        --bench results/baselines/*_bench.json results/jetson/*/benchmark.json \
        --accuracy-runtime engine --latency-runtime engine --batch 1 \
        --output results/baselines/comparison
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

MARKERS = {"official": "", "reimplemented": r"$^{\dagger}$", "reported": r"$^{\ddagger}$"}
SEED = re.compile(r"_seed\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", type=Path, nargs="+", required=True)
    parser.add_argument("--bench", type=Path, nargs="*", default=[])
    parser.add_argument("--accuracy-runtime", default="pt", help="pt, onnx or engine")
    parser.add_argument("--latency-runtime", default="engine")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True, help="path prefix; writes .tex and .csv")
    return parser.parse_args()


def group(name: str) -> str:
    return SEED.sub("", name)


def load_eval(paths: list[Path], runtime: str) -> dict[str, dict]:
    rows: dict[str, dict] = defaultdict(lambda: {"runs": [], "source": "official", "family": ""})
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        baseline = payload.get("baseline")
        for model_name, model in payload["models"].items():
            if baseline:  # evaluate_baseline.py output
                if model.get("runtime") != runtime:
                    continue
                key, source, family = group(baseline["name"]), baseline.get("source", "official"), baseline["family"]
            else:  # evaluate_deployments.py output (student + teachers)
                suffix = {"pt": "_pt", "onnx": "_onnx", "engine": "_engine"}[runtime]
                if model.get("family") == "student" and not model_name.endswith(suffix):
                    continue
                key, source, family = model_name.replace(suffix, ""), "official", model.get("family", "")
            domains = model["domains"]
            rows[key]["runs"].append(
                {
                    "cv": domains.get("cv", {}).get("metrics", {}).get("macro_f1_present_classes"),
                    "rs": domains.get("rs", {}).get("metrics", {}).get("macro_f1_present_classes"),
                    "uav": domains.get("uav", {}).get("metrics", {}).get("macro_f1_present_classes"),
                    "acc": model["domain_macro"]["accuracy"],
                    "f1": model["domain_macro"]["macro_f1_present_classes"],
                }
            )
            rows[key]["source"], rows[key]["family"] = source, family
    return rows


def load_bench(paths: list[Path], runtime: str, batch: int) -> dict[str, dict]:
    rows: dict[str, dict] = defaultdict(lambda: {"runs": []})
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        # benchmark_deployments.py (student) has no "baseline" block; its rows are named "student".
        name = payload.get("baseline", {}).get("name") or ("student" if "model" in payload else path.stem)
        params = payload.get("baseline", {}).get("parameter_count") or payload.get("model", {}).get("parameter_count")
        runtimes = payload.get("runtimes", {})
        entry = runtimes.get(runtime) or runtimes.get(f"student_{runtime}")
        if not entry or str(batch) not in entry["batches"]:
            continue
        b = entry["batches"][str(batch)]
        rows[group(name)]["runs"].append(
            {
                "ms": b["mean_ms_per_image"],
                "p99": b.get("p99_batch_ms"),
                "fps": b["throughput_images_per_second"],
                "mj": b.get("energy_mj_per_image"),
                "params": params,
            }
        )
    return rows


def stat(values: list, fmt: str, scale: float = 1.0) -> str:
    values = [v * scale for v in values if v is not None]
    if not values:
        return "--"
    if len(values) == 1:
        return fmt.format(values[0])
    return (fmt + r" $\pm$ " + fmt).format(float(np.mean(values)), float(np.std(values, ddof=1)))


def main() -> None:
    args = parse_args()
    accuracy = load_eval(args.eval, args.accuracy_runtime)
    speed = load_bench(args.bench, args.latency_runtime, args.batch)

    header = ["Method", "CV F1", "RS F1", "UAV F1", "Acc", "Macro-F1", "Params (M)", "ms/img", "FPS", "mJ/img"]
    tex_rows, csv_rows = [], []
    for key in sorted(accuracy, key=lambda k: -np.mean([r["f1"] for r in accuracy[k]["runs"]])):
        runs = accuracy[key]["runs"]
        bench = speed.get(key, {"runs": []})["runs"]
        cells = [
            key.replace("_", r"\_") + MARKERS.get(accuracy[key]["source"], ""),
            stat([r["cv"] for r in runs], "{:.1f}", 100),
            stat([r["rs"] for r in runs], "{:.1f}", 100),
            stat([r["uav"] for r in runs], "{:.1f}", 100),
            stat([r["acc"] for r in runs], "{:.1f}", 100),
            stat([r["f1"] for r in runs], "{:.1f}", 100),
            stat([b["params"] for b in bench[:1]], "{:.1f}", 1e-6),
            stat([b["ms"] for b in bench], "{:.2f}"),
            stat([b["fps"] for b in bench], "{:.1f}"),
            stat([b["mj"] for b in bench], "{:.1f}"),
        ]
        tex_rows.append(" & ".join(cells) + r" \\")
        csv_rows.append([key, accuracy[key]["source"], accuracy[key]["family"], len(runs)] + [
            re.sub(r"\$|\\pm|\\dagger|\\ddagger|\^|\{|\}|\\", "", c).strip() for c in cells[1:]
        ])

    tex = "\n".join(
        [
            r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}",
            r"\toprule",
            " & ".join(header) + r" \\",
            r"\midrule",
            *tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"% F1 = macro-F1 over classes present in each domain's test split (%).",
            f"% Accuracy runtime: {args.accuracy_runtime}; latency runtime: {args.latency_runtime}, batch {args.batch}.",
            r"% $^{\dagger}$ reimplemented from the paper; $^{\ddagger}$ reported by the authors.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".tex").write_text(tex + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "source", "family", "seeds"] + header[1:])
        writer.writerows(csv_rows)
    print(tex)


if __name__ == "__main__":
    main()
