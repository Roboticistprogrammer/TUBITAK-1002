"""Regenerate the thesis figures from the final results (the placeholders in Sections/Results.tex).

    python scripts/pipeline/10_make_figures.py --stats results/statistics/statistics.json \
        --history-root outputs/retrain --jetson-root results/jetson --out-dir figures

Writes (copy into the thesis `Figures/` and replace each `\\figplaceholder` by `\\includegraphics`):
  domain_comparison.png        accuracy per domain, mean +- std over seeds        (needs --stats)
  training_overview.png        validation accuracy per epoch, mean +- std         (needs --history-root)
  latency_percentiles.png      p50/p95/p99 latency per runtime, batch 1           (needs --jetson-root)
  throughput.png               images/s per runtime and batch size                (needs --jetson-root)
  deployment_comparison.png    latency, peak RAM footprint and model size         (needs --jetson-root; sizes from the benchmark JSONs)
Inputs that are missing are skipped with a message, so the script can be run as results arrive.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GROUP_LABELS = {"teacher_ensemble": "Teachers (domain-matched)", "kd_uniform": "KD, uniform", "kd_routed": "KD, domain-routed", "nokd": "No KD"}
RUNTIME_LABELS = {"pytorch": "PyTorch", "onnx": "ONNX Runtime", "trt_fp16": "TensorRT FP16", "trt_int8": "TensorRT INT8"}
DOMAIN_ORDER = ["cv", "uav", "rs"]
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def domain_comparison(stats_path: Path, out: Path) -> None:
    """Dot-and-whisker (mean +- std over seeds) so the accuracy axis need not start at zero."""
    groups = json.loads(stats_path.read_text(encoding="utf-8"))["groups"]
    names = [g for g in GROUP_LABELS if g in groups]
    fig, ax = plt.subplots(figsize=(7, 4))
    spread = 0.5 / max(1, len(names))
    for i, group in enumerate(names):
        means = [100 * groups[group]["domains"][d]["accuracy"]["mean"] for d in DOMAIN_ORDER]
        stds = [100 * (groups[group]["domains"][d]["accuracy"]["std"] or 0) for d in DOMAIN_ORDER]
        x = np.arange(3) + (i - (len(names) - 1) / 2) * spread
        ax.errorbar(x, means, yerr=stds, fmt="o", capsize=3, label=GROUP_LABELS[group])
    ax.set_xticks(np.arange(3), [d.upper() for d in DOMAIN_ORDER])
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=len(names))
    fig.tight_layout(); fig.savefig(out / "domain_comparison.png"); plt.close(fig)


def training_overview(history_root: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    drawn = False
    for group in ("kd_uniform", "kd_routed", "nokd"):
        curves = []
        for path in sorted((history_root / group).glob("seed*/history.json")):
            records = json.loads(path.read_text(encoding="utf-8"))["records"]
            curves.append([100 * r["avg_val"] for r in records])
        if not curves:
            continue
        length = min(map(len, curves))
        data = np.asarray([c[:length] for c in curves])
        epochs = np.arange(1, length + 1)
        ax.plot(epochs, data.mean(0), label=f"{GROUP_LABELS[group]} (n={len(curves)})")
        if len(curves) > 1:
            ax.fill_between(epochs, data.mean(0) - data.std(0, ddof=1), data.mean(0) + data.std(0, ddof=1), alpha=0.2)
        drawn = True
    if not drawn:
        plt.close(fig); print("training_overview: no history found"); return
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean validation accuracy over domains (%)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "training_overview.png"); plt.close(fig)


def load_benchmarks(jetson_root: Path, name: str | None) -> dict[str, dict]:
    """runtime label -> {'batches': {...}, 'artifact_mib': ..., 'memory idle': ...} from <name>_<runtime>.json."""
    found = {}
    for path in sorted(jetson_root.glob("*.json")):
        match = re.match(r"(.+)_(pytorch|onnx|trt_fp16|trt_int8)\.json$", path.name)
        if not match or (name and match.group(1) != name):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        runtime = next(iter(payload["runtimes"].values()))
        found[match.group(2)] = {"batches": runtime["batches"], "artifact_mib": runtime["artifact_mib"],
                                 "idle_ram": payload["environment"].get("idle_ram_used_mb")}
    return found


def latency_figures(jetson_root: Path, name: str | None, out: Path) -> None:
    bench = load_benchmarks(jetson_root, name)
    if not bench:
        print("latency/throughput/deployment: no <name>_<runtime>.json under", jetson_root); return
    runtimes = [r for r in RUNTIME_LABELS if r in bench]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, pct in enumerate(("p50", "p95", "p99")):
        values = [bench[r]["batches"]["1"][f"{pct}_batch_ms"] for r in runtimes]
        ax.bar(np.arange(len(runtimes)) + i * 0.26, values, 0.26, label=pct)
    ax.set_xticks(np.arange(len(runtimes)) + 0.26, [RUNTIME_LABELS[r] for r in runtimes])
    ax.set_ylabel("Latency, batch 1 (ms)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "latency_percentiles.png"); plt.close(fig)

    batches = sorted({int(b) for r in runtimes for b in bench[r]["batches"]})
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, batch in enumerate(batches):
        values = [bench[r]["batches"].get(str(batch), {}).get("throughput_images_per_second", 0) for r in runtimes]
        ax.bar(np.arange(len(runtimes)) + i * 0.8 / len(batches), values, 0.8 / len(batches), label=f"batch {batch}")
    ax.set_xticks(np.arange(len(runtimes)) + 0.4 - 0.4 / len(batches), [RUNTIME_LABELS[r] for r in runtimes])
    ax.set_ylabel("Throughput (images/s)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "throughput.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    labels = [RUNTIME_LABELS[r] for r in runtimes]
    axes[0].bar(labels, [bench[r]["batches"]["1"]["mean_batch_ms"] for r in runtimes]); axes[0].set_title("Latency, batch 1 (ms)")
    footprint = []
    for r in runtimes:
        memory = bench[r]["batches"]["1"].get("memory")
        idle = bench[r]["idle_ram"]
        footprint.append(memory["peak_ram_mb"] - idle if memory and idle else 0)
    axes[1].bar(labels, footprint); axes[1].set_title("Peak RAM above idle (MB)")
    axes[2].bar(labels, [bench[r]["artifact_mib"] for r in runtimes]); axes[2].set_title("Artifact size (MiB)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(out / "deployment_comparison.png"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", type=Path, default=Path("results/statistics/statistics.json"))
    parser.add_argument("--history-root", type=Path, default=Path("outputs/retrain"))
    parser.add_argument("--jetson-root", type=Path, default=Path("results/jetson"))
    parser.add_argument("--name", help="model whose Jetson benchmarks to plot, e.g. kd_uniform_seed42 (default: all found)")
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stats.exists():
        domain_comparison(args.stats, args.out_dir)
    else:
        print("domain_comparison: missing", args.stats)
    training_overview(args.history_root, args.out_dir)
    latency_figures(args.jetson_root, args.name, args.out_dir)
    print("Figures in", args.out_dir, sorted(p.name for p in args.out_dir.glob("*.png")))


if __name__ == "__main__":
    main()
