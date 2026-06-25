from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot deployment latency and throughput benchmarks.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/deployment_benchmark.png"))
    return parser.parse_args()


def runtime_label(name: str) -> str:
    return name.replace("student_", "").upper().replace("_", " ")


def grouped_bars(axis, batch_sizes: list[str], runtime_names: list[str], values: dict, ylabel: str) -> None:
    x = np.arange(len(batch_sizes))
    width = 0.78 / max(len(runtime_names), 1)
    offsets = (np.arange(len(runtime_names)) - (len(runtime_names) - 1) / 2) * width

    for offset, runtime_name in zip(offsets, runtime_names):
        series = [values[runtime_name][batch_size] for batch_size in batch_sizes]
        bars = axis.bar(x + offset, series, width=width, label=runtime_label(runtime_name))
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

    axis.set_xticks(x, [f"batch {batch_size}" for batch_size in batch_sizes])
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    runtimes = payload["runtimes"]
    runtime_names = list(runtimes)
    batch_sizes = sorted(
        {batch_size for runtime in runtimes.values() for batch_size in runtime["batches"]},
        key=int,
    )

    latency = {
        name: {
            batch_size: runtimes[name]["batches"][batch_size]["mean_ms_per_image"]
            for batch_size in batch_sizes
            if batch_size in runtimes[name]["batches"]
        }
        for name in runtime_names
    }
    throughput = {
        name: {
            batch_size: runtimes[name]["batches"][batch_size]["throughput_images_per_second"]
            for batch_size in batch_sizes
            if batch_size in runtimes[name]["batches"]
        }
        for name in runtime_names
    }
    artifact_sizes = [runtimes[name]["artifact_mib"] for name in runtime_names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    grouped_bars(axes[0], batch_sizes, runtime_names, latency, "Mean latency per image (ms)")
    axes[0].set_title("Lower is better")

    grouped_bars(axes[1], batch_sizes, runtime_names, throughput, "Throughput (images/s)")
    axes[1].set_title("Higher is better")

    bars = axes[2].bar([runtime_label(name) for name in runtime_names], artifact_sizes)
    axes[2].bar_label(bars, fmt="%.1f MiB", padding=3, fontsize=8)
    axes[2].set_ylabel("Artifact size (MiB)")
    axes[2].set_title("Deployment artifact size")
    axes[2].grid(axis="y", alpha=0.25)

    environment = payload.get("environment", {})
    model = payload.get("model", {})
    device = environment.get("device", "unknown device")
    model_name = model.get("model_name", "student model")
    fig.suptitle(f"Deployment benchmark: {model_name}\n{device}", weight="bold")
    fig.legend(loc="lower center", ncol=max(len(runtime_names), 1), bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
