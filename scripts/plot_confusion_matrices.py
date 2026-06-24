from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot row-normalized confusion matrices.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/confusion_matrices.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    classes = payload["protocol"]["classes"]
    domains = payload["protocol"]["domains"]
    model_names = list(payload["models"])
    available = [
        (domain, model)
        for domain in domains
        for model in model_names
        if domain in payload["models"][model]["domains"]
    ]
    columns = max(sum(1 for d, _ in available if d == domain) for domain in domains)
    fig, axes = plt.subplots(
        len(domains), columns, figsize=(4.2 * columns, 4 * len(domains)), squeeze=False
    )

    for row, domain in enumerate(domains):
        domain_models = [model for d, model in available if d == domain]
        for column in range(columns):
            axis = axes[row, column]
            if column >= len(domain_models):
                axis.axis("off")
                continue
            model = domain_models[column]
            metrics = payload["models"][model]["domains"][domain]["metrics"]
            normalized = np.asarray(metrics["confusion_matrix_normalized"])
            counts = np.asarray(metrics["confusion_matrix"])
            image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
            for y in range(len(classes)):
                for x in range(len(classes)):
                    color = "white" if normalized[y, x] > 0.55 else "black"
                    axis.text(
                        x,
                        y,
                        f"{counts[y, x]}\n{normalized[y, x]:.1%}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                    )
            axis.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
            axis.set_yticks(range(len(classes)), classes)
            axis.set_xlabel("Predicted class")
            axis.set_ylabel("True class")
            axis.set_title(
                f"{domain.upper()} — {model}\n"
                f"Acc {metrics['accuracy']:.3f} | Macro-F1 {metrics['macro_f1_present_classes']:.3f}"
            )
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02, label="Row proportion")
    fig.suptitle("Teacher–Student and Deployment Fidelity on Held-Out Test Sets", weight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
