"""Tabulate results/jetson/*.json (from pipeline/05b) into one CSV + Markdown table.

    python scripts/summarize_jetson_benchmarks.py results/jetson/*.json --output results/test_dataset_eval
Columns: model, runtime (ort_cpu / ort_cuda / trt_fp16), batch, p50 / p95 latency (ms per batch), throughput, mean power, mJ per image, peak RAM.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RUNTIMES = ("ort_cpu", "ort_cuda", "trt_fp16")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/test_dataset_eval"))
    args = parser.parse_args()
    rows = []
    for path in sorted(args.files):
        if path.suffix != ".json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        name = data["baseline"]["name"]
        runtime = next((r for r in RUNTIMES if name.endswith("_" + r)), "?")
        model = name[: -len(runtime) - 1] if runtime != "?" else name
        for rt in data["runtimes"].values():
            for b in rt["batches"].values():
                rows.append({
                    "model": model, "runtime": runtime, "batch": b["batch_size"],
                    "p50_ms": b["p50_batch_ms"], "p95_ms": b["p95_batch_ms"],
                    "throughput_img_s": b["throughput_images_per_second"],
                    "power_mw": b.get("power", {}).get("mean_mw"),
                    "energy_mj_per_img": b.get("energy_mj_per_image"),
                    "peak_ram_mb": b.get("memory", {}).get("peak_ram_mb"),
                })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "latency_throughput.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
    lines = ["| model | runtime | batch | p50 ms/batch | p95 ms/batch | img/s | power mW | mJ/img | peak RAM MB |", "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['runtime']} | {r['batch']} | {fmt(r['p50_ms'])} | {fmt(r['p95_ms'])} | {fmt(r['throughput_img_s'])} | {fmt(r['power_mw'], 0)} | {fmt(r['energy_mj_per_img'], 0)} | {fmt(r['peak_ram_mb'], 0)} |" for r in rows]
    (args.output / "latency_throughput.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
