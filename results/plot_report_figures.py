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
    size = [deployment_of(m, "size_mb") for m in models]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(14, 6))

    bar1 = ax1.bar(x - width / 2, latency, width, label="Latency (ms)", color="#FF006E")
    bar2 = ax1.bar(x + width / 2, fps, width, label="FPS", color="#3A86FF")

    ax1.set_title("Deployment Performance (Host Machine)", fontsize=16, pad=16, weight="bold")
    ax1.set_ylabel("Latency / FPS")
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
    ax2.plot(x, size, color="#06D6A0", marker="o", linewidth=2, label="Model Size (MB)")
    ax2.set_ylabel("Model Size (MB)")

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

    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(grid[0, 0])
    ax1.bar(model_names, m_ap50, color="#3A86FF")
    ax1.set_title("mAP@0.5 (%)", weight="bold")
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="x", rotation=10)
    ax1.grid(axis="y", alpha=0.2)

    ax2 = fig.add_subplot(grid[0, 1])
    ax2.bar(model_names, recall, color="#06D6A0")
    ax2.set_title("Recall (%)", weight="bold")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="x", rotation=10)
    ax2.grid(axis="y", alpha=0.2)

    ax3 = fig.add_subplot(grid[1, 0])
    ax3.bar(model_names, f1_score, color="#FFB703")
    ax3.set_title("F1 (%)", weight="bold")
    ax3.set_ylim(0, 100)
    ax3.tick_params(axis="x", rotation=10)
    ax3.grid(axis="y", alpha=0.2)

    ax4 = fig.add_subplot(grid[1, 1])
    ax4.bar(model_names, latency, color="#FF006E")
    ax4.set_title("Latency (ms)", weight="bold")
    ax4.tick_params(axis="x", rotation=10)
    ax4.grid(axis="y", alpha=0.2)

    fig.suptitle("Teacher vs Student Report Dashboard", fontsize=18, weight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_dir / "figure_dashboard.png", dpi=300)
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

    print(f"Saved figures to: {args.output_dir}")
    print("- figure_accuracy_comparison.png")
    print("- figure_domain_demographics.png (if domain data exists)")
    print("- figure_deployment_comparison.png")
    print("- figure_dashboard.png")


if __name__ == "__main__":
    main()
