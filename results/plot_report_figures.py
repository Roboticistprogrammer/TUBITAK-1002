import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_COLORS = {
    "teacher": "#5B5F97",
    "student": "#1EA896",
    "pt": "#3A86FF",
    "onnx": "#8338EC",
    "engine": "#FFBE0B",
}


def load_benchmark_data(input_path: Path):
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "models" not in data:
        raise ValueError("Input JSON must contain a 'models' list field.")

    return data


def ensure_output_dir(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)


def metric_of(model, key, default=0.0):
    return float(model.get("metrics", {}).get(key, default))


def deployment_of(model, key, default=0.0):
    return float(model.get("deployment", {}).get(key, default))


def save_accuracy_figure(models, output_dir: Path):
    model_names = [m["name"] for m in models]
    m_ap50 = [metric_of(m, "mAP50") * 100 for m in models]
    recall = [metric_of(m, "recall") * 100 for m in models]
    f1_score = [metric_of(m, "f1") * 100 for m in models]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars_1 = ax.bar(x - width, m_ap50, width, label="mAP@0.5", color="#3A86FF")
    bars_2 = ax.bar(x, recall, width, label="Recall", color="#06D6A0")
    bars_3 = ax.bar(x + width, f1_score, width, label="F1", color="#FFB703")

    ax.set_title("Model Accuracy Comparison", fontsize=16, pad=16, weight="bold")
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=10, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    for bars in [bars_1, bars_2, bars_3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_dir / "figure_accuracy_comparison.png", dpi=300)
    plt.close(fig)


def save_domain_figure(models, output_dir: Path):
    all_domains = sorted(
        {
            domain
            for model in models
            for domain in model.get("domains", {}).keys()
        }
    )
    if not all_domains:
        return

    model_names = [m["name"] for m in models]
    domain_values = {
        domain: [
            float(model.get("domains", {}).get(domain, {}).get("mAP50", 0.0)) * 100
            for model in models
        ]
        for domain in all_domains
    }

    x = np.arange(len(model_names))
    width = 0.8 / max(len(all_domains), 1)

    fig, ax = plt.subplots(figsize=(14, 6))
    palette = ["#118AB2", "#EF476F", "#06D6A0", "#FFD166", "#8338EC"]

    for idx, domain in enumerate(all_domains):
        values = domain_values[domain]
        offset = (idx - (len(all_domains) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=domain, color=palette[idx % len(palette)])

    ax.set_title("Per-Domain mAP@0.5 (Dataset Demographics)", fontsize=16, pad=16, weight="bold")
    ax.set_ylabel("mAP@0.5 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=10, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=min(3, len(all_domains)))

    fig.tight_layout()
    fig.savefig(output_dir / "figure_domain_demographics.png", dpi=300)
    plt.close(fig)


def save_deployment_figure(models, output_dir: Path):
    model_names = [m["name"] for m in models]
    latency = [deployment_of(m, "latency_ms") for m in models]
    fps = [deployment_of(m, "fps") for m in models]
    # weights_mb preferred; fall back to size_mb for backward compat
    weights = [
        float(m.get("deployment", {}).get("weights_mb") or m.get("deployment", {}).get("size_mb", 0.0))
        for m in models
    ]
    power = [deployment_of(m, "power_draw_w") for m in models]
    has_power = any(p > 0 for p in power)

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(14, 6))

    bar1 = ax1.bar(x - width / 2, latency, width, label="Latency (ms)", color="#FF006E")
    bar2 = ax1.bar(x + width / 2, fps, width, label="FPS", color="#3A86FF")

    ax1.set_title("Deployment Performance (Host Machine)", fontsize=16, pad=16, weight="bold")
    ax1.set_ylabel("Latency (ms) / FPS")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=10, ha="right")
    ax1.grid(axis="y", alpha=0.2)

    for bars in [bar1, bar2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax2 = ax1.twinx()
    ax2.plot(x, weights, color="#06D6A0", marker="o", linewidth=2, label="Weights (MB)")
    if has_power:
        ax2.plot(x, power, color="#FB5607", marker="s", linewidth=2, linestyle="--", label="Power draw (W)")
    ax2.set_ylabel("Weights (MB) / Power (W)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "figure_deployment_comparison.png", dpi=300)
    plt.close(fig)


def save_dashboard(models, output_dir: Path):
    model_names = [m["name"] for m in models]
    m_ap50 = [metric_of(m, "mAP50") * 100 for m in models]
    recall = [metric_of(m, "recall") * 100 for m in models]
    f1_score = [metric_of(m, "f1") * 100 for m in models]
    latency = [deployment_of(m, "latency_ms") for m in models]
    fps = [deployment_of(m, "fps") for m in models]
    power = [deployment_of(m, "power_draw_w") for m in models]
    weights = [
        float(m.get("deployment", {}).get("weights_mb") or m.get("deployment", {}).get("size_mb", 0.0))
        for m in models
    ]
    has_power = any(p > 0 for p in power)

    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # Row 0 – accuracy metrics
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.bar(model_names, m_ap50, color="#3A86FF")
    ax1.set_title("mAP@0.5 (%)", weight="bold")
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="x", rotation=15)
    ax1.grid(axis="y", alpha=0.2)

    ax2 = fig.add_subplot(grid[0, 1])
    ax2.bar(model_names, recall, color="#06D6A0")
    ax2.set_title("Recall (%)", weight="bold")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(axis="y", alpha=0.2)

    ax3 = fig.add_subplot(grid[0, 2])
    ax3.bar(model_names, f1_score, color="#FFB703")
    ax3.set_title("F1 (%)", weight="bold")
    ax3.set_ylim(0, 100)
    ax3.tick_params(axis="x", rotation=15)
    ax3.grid(axis="y", alpha=0.2)

    # Row 1 – deployment metrics
    ax4 = fig.add_subplot(grid[1, 0])
    ax4.bar(model_names, latency, color="#FF006E")
    ax4.set_title("Latency (ms)", weight="bold")
    ax4.tick_params(axis="x", rotation=15)
    ax4.grid(axis="y", alpha=0.2)

    ax5 = fig.add_subplot(grid[1, 1])
    if has_power:
        ax5.bar(model_names, power, color="#FB5607")
        ax5.set_title("Power Draw (W)", weight="bold")
    else:
        ax5.bar(model_names, fps, color="#8338EC")
        ax5.set_title("FPS", weight="bold")
    ax5.tick_params(axis="x", rotation=15)
    ax5.grid(axis="y", alpha=0.2)

    ax6 = fig.add_subplot(grid[1, 2])
    ax6.bar(model_names, weights, color="#118AB2")
    ax6.set_title("Model Weights (MB)", weight="bold")
    ax6.tick_params(axis="x", rotation=15)
    ax6.grid(axis="y", alpha=0.2)

    fig.suptitle("Teacher vs Student Report Dashboard", fontsize=18, weight="bold")
    fig.savefig(output_dir / "figure_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_format_comparison_figure(models, output_dir: Path):
    """
    Four-panel deployment comparison figure styled for thesis inclusion.
    Compares PT vs ONNX vs TensorRT Engine across:
      Max Throughput (samples/sec) | Latency @ batch=1 (ms)
      Avg GPU Power Draw (W)       | Model Weights on Disk (MB)
    Each format gets a distinct color + hatch pattern for B&W printability.
    """
    FORMAT_HATCHES = {"pt": "", "onnx": "///", "engine": "xxx"}

    names       = [m["name"] for m in models]
    fmts        = [m.get("format", "pt") for m in models]
    bar_colors  = [DEFAULT_COLORS.get(f, "#607D8B") for f in fmts]
    bar_hatches = [FORMAT_HATCHES.get(f, "") for f in fmts]

    throughput = [
        float(
            m.get("deployment", {}).get("throughput_samples_per_sec")
            or m.get("deployment", {}).get("fps", 0.0)
        )
        for m in models
    ]
    latency = [deployment_of(m, "latency_ms") for m in models]
    power   = [deployment_of(m, "power_draw_w") for m in models]
    weights = [
        float(
            m.get("deployment", {}).get("weights_mb")
            or m.get("deployment", {}).get("size_mb", 0.0)
        )
        for m in models
    ]
    has_power = any(p > 0 for p in power)
    x = np.arange(len(names))
    w = 0.55

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Student Model Format Comparison: PT vs ONNX vs TensorRT Engine",
        fontsize=14, weight="bold",
    )

    def _barplot(ax, values, ylabel, title, fmt_str="{:.0f}"):
        for i, (val, color, hatch) in enumerate(zip(values, bar_colors, bar_hatches)):
            ax.bar(i, val, w, color=color, hatch=hatch, edgecolor="white", linewidth=0.8)
            ax.annotate(
                fmt_str.format(val),
                xy=(i, val), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=9, weight="bold",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, weight="bold", pad=8)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    _barplot(axes[0, 0], throughput, "Samples / sec", "Max Throughput  \u2191 better", "{:.0f}")
    _barplot(axes[0, 1], latency,    "ms / image",    "Latency @ batch=1  \u2193 better", "{:.1f}")

    if has_power:
        _barplot(axes[1, 0], power, "Watts", "Avg GPU Power Draw  \u2193 better", "{:.1f}")
    else:
        axes[1, 0].text(
            0.5, 0.5,
            "Power data unavailable\n(run with nvidia-ml-py installed)",
            ha="center", va="center", transform=axes[1, 0].transAxes,
            fontsize=10, color="#888",
        )
        axes[1, 0].set_title("Avg GPU Power Draw (W)  \u2193 better", weight="bold", pad=8)
        axes[1, 0].spines[["top", "right"]].set_visible(False)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(names, rotation=12, ha="right")

    _barplot(axes[1, 1], weights, "MB", "Model Weights on Disk", "{:.1f}")

    # Format legend with color + hatch
    from matplotlib.patches import Patch
    handles = [
        Patch(
            facecolor=DEFAULT_COLORS.get(f, "#607D8B"),
            hatch=FORMAT_HATCHES.get(f, ""),
            edgecolor="white", label=n,
        )
        for n, f in zip(names, fmts)
    ]
    fig.legend(
        handles=handles, loc="lower center", ncols=len(models),
        frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_dir / "figure_format_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate colorful benchmark figures from JSON input.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/benchmark_metrics.json"),
        help="Path to benchmark metrics JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory where PNG figures are stored.",
    )
    args = parser.parse_args()

    data = load_benchmark_data(args.input)
    ensure_output_dir(args.output_dir)
    models = data.get("models", [])

    if not models:
        print("No models found in input JSON. Skipping figure generation.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    save_accuracy_figure(models, args.output_dir)
    save_domain_figure(models, args.output_dir)
    save_deployment_figure(models, args.output_dir)
    save_dashboard(models, args.output_dir)
    save_format_comparison_figure(models, args.output_dir)

    print(f"Saved figures to: {args.output_dir}")
    print("- figure_accuracy_comparison.png")
    print("- figure_domain_demographics.png (if domain data exists)")
    print("- figure_deployment_comparison.png")
    print("- figure_dashboard.png")
    print("- figure_format_comparison.png  <-- PT vs ONNX vs Engine deployment")


if __name__ == "__main__":
    main()
