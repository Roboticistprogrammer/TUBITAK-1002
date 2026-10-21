"""Cross-architecture comparison figures: MobileNetV3-Large, ResNet-50 and the three SwinV2-B students.

    python scripts/plot_cross_architecture.py

Reads the per-image logits in ``<arch>/predictions/<model>_trt_fp16/{cv,uav,rs}.npz`` and the Jetson
benchmarks in ``<arch>/benchmarks/``; writes PDF (vector) + PNG (300 dpi) figures, a summary CSV and a
Markdown table into ``comparison/``. Accuracy uses the TensorRT FP16 engine (the deployed model).
"Macro accuracy" is the equal-weight mean of the CV, UAV and RS accuracies (as in the per-model READMEs).

Colour rule: scatter plots (all pairs of series visible at once) colour by architecture family, three
hues, which is what the validated palette supports; the three SwinV2-B variants share a hue and are told
apart by direct labels. Bar charts colour by runtime (same hues as the per-model runtime figure).
Latency/energy/throughput are Jetson Orin numbers from random 192x192 input, preprocessing excluded.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "comparison"
DOMAINS = ["cv", "uav", "rs"]
DOMAIN_LABEL = {"cv": "CV (ground)", "uav": "UAV", "rs": "RS (satellite)"}

# (model id, arch dir, display name, family, params in millions)
MODELS = [
    ("mobilenet_v3_large_seed42", "mobilenet_v3_large", "MobileNetV3-Large", "mobilenet", 4.2),
    ("resnet50_seed42", "resnet50", "ResNet-50", "resnet", 23.5),
    ("kd_routed_seed42", "swinv2-b-students", "SwinV2-B KD routed", "swin", 86.9),
    ("kd_uniform_seed42", "swinv2-b-students", "SwinV2-B KD uniform", "swin", 86.9),
    ("nokd_seed42", "swinv2-b-students", "SwinV2-B no KD", "swin", 86.9),
]
SHORT = {"kd_routed_seed42": "KD routed", "kd_uniform_seed42": "KD uniform", "nokd_seed42": "no KD",
         "mobilenet_v3_large_seed42": "MobileNetV3-L", "resnet50_seed42": "ResNet-50"}
# Validated categorical palette (light surface); slots 1-3 validate for all-pairs scatter.
FAMILY_COLOR = {"mobilenet": "#2a78d6", "resnet": "#eb6834", "swin": "#1baf7a"}
FAMILY_LABEL = {"mobilenet": "MobileNetV3-Large (4.2M)", "resnet": "ResNet-50 (23.5M)", "swin": "SwinV2-B students (86.9M)"}
MARKER = {"mobilenet_v3_large_seed42": "o", "resnet50_seed42": "o", "kd_routed_seed42": "o",
          "kd_uniform_seed42": "s", "nokd_seed42": "^"}
RUNTIMES = [("ort_cpu", "ORT CPU", "#2a78d6"), ("ort_cuda", "ORT CUDA", "#eb6834"), ("trt_fp16", "TensorRT FP16", "#1baf7a")]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.labelsize": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "legend.fontsize": 8, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
})


# ----------------------------------------------------------------------------- data
def accuracy(model: str, arch: str) -> dict:
    out = {}
    for d in DOMAINS:
        z = np.load(ROOT / arch / "predictions" / f"{model}_trt_fp16" / f"{d}.npz", allow_pickle=False)
        correct = int((z["logits"].argmax(1) == z["labels"]).sum())
        out[d] = (correct, len(z["labels"]))
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def bench(arch: str, model: str, runtime: str) -> dict:
    data = json.loads((ROOT / arch / "benchmarks" / f"{model}_{runtime}.json").read_text(encoding="utf-8"))
    rt = next(iter(data["runtimes"].values()))
    return {"batches": rt["batches"], "artifact_mib": rt["artifact_mib"]}


def collect() -> list[dict]:
    rows = []
    for model, arch, name, family, params in MODELS:
        acc = accuracy(model, arch)
        row = {"model": model, "name": name, "family": family, "params_m": params,
               "acc": {d: acc[d][0] / acc[d][1] for d in DOMAINS},
               "ci": {d: wilson(*acc[d]) for d in DOMAINS}, "n": {d: acc[d][1] for d in DOMAINS}}
        row["macro_acc"] = float(np.mean(list(row["acc"].values())))
        row["bench"] = {}
        for rt, _, _ in RUNTIMES:
            b = bench(arch, model, rt)
            row["bench"][rt] = b["batches"]
            if rt == "trt_fp16":
                row["engine_mib"] = b["artifact_mib"]
        rows.append(row)
    return rows


def b1(row, rt, key):
    return row["bench"][rt]["1"][key]


# ----------------------------------------------------------------------------- helpers
def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figures" / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def style_axes(ax, ygrid=True):
    if ygrid:
        ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def family_legend(fig, y=0.97, shapes=False):
    """Family colours; with shapes=True the SwinV2-B entry is replaced by its three variants (same hue, own marker)."""
    kw = dict(ls="", markeredgecolor="white", markeredgewidth=1.2)
    handles = [plt.Line2D([], [], marker="o", color=FAMILY_COLOR[f], markersize=7, label=FAMILY_LABEL[f], **kw)
               for f in ("mobilenet", "resnet")]
    if shapes:
        handles += [plt.Line2D([], [], marker=MARKER[k], color=FAMILY_COLOR["swin"], markersize=7, label=f"SwinV2-B {SHORT[k]} (86.9M)", **kw)
                    for k in ("kd_routed_seed42", "kd_uniform_seed42", "nokd_seed42")]
    else:
        handles.append(plt.Line2D([], [], marker="o", color=FAMILY_COLOR["swin"], markersize=7, label=FAMILY_LABEL["swin"], **kw))
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, y), columnspacing=2.4)


OFFSET = {  # label offsets (points) so the three near-identical SwinV2 variants do not collide
    "kd_routed_seed42": (7, 6), "kd_uniform_seed42": (8, 5), "nokd_seed42": (8, -8),
    "mobilenet_v3_large_seed42": (7, 5), "resnet50_seed42": (7, 5),
}


XJIT = {"kd_routed_seed42": 0.93, "kd_uniform_seed42": 1.0, "nokd_seed42": 1.075}  # fig2 only: separates the 3 variants


def scatter_points(ax, rows, xfun, yfun, yerr=None, labels=True, jitter=False):
    for r in rows:
        x, y = xfun(r), yfun(r)
        if jitter:
            x *= XJIT.get(r["model"], 1.0)
        if yerr is not None:
            lo, hi = yerr(r)
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=FAMILY_COLOR[r["family"]], elinewidth=1.1,
                        capsize=2.5, zorder=2)
        ax.scatter([x], [y], s=44, marker=MARKER[r["model"]], color=FAMILY_COLOR[r["family"]], edgecolor="white",
                   linewidth=1.3, zorder=3)
        if labels:
            ax.annotate(SHORT[r["model"]], (x, y), xytext=OFFSET[r["model"]], textcoords="offset points",
                        fontsize=7.5, color=INK2, va="center")


# ----------------------------------------------------------------------------- figures
def fig_accuracy_vs_cost(rows):
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), sharey=True)
    specs = [
        (lambda r: b1(r, "trt_fp16", "p50_batch_ms"), "Latency, batch 1, p50 (ms, log)", True, "Accuracy vs latency"),
        (lambda r: b1(r, "trt_fp16", "energy_mj_per_image") if "energy_mj_per_image" in r["bench"]["trt_fp16"]["1"] else np.nan,
         "Energy per image, batch 1 (mJ, log)", True, "Accuracy vs energy"),
        (lambda r: r["engine_mib"], "TensorRT engine size (MiB, log)", True, "Accuracy vs engine size"),
    ]
    for ax, (xf, xlabel, log, title) in zip(axes, specs):
        scatter_points(ax, rows, xf, lambda r: 100 * r["macro_acc"])
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        style_axes(ax)
        xs = np.array([xf(r) for r in rows])
        ax.set_xlim(xs.min() / 1.5, xs.max() * 2.6)
    axes[0].set_ylabel("Macro accuracy over CV, UAV, RS (%)")
    ys = np.array([100 * r["macro_acc"] for r in rows])
    axes[0].set_ylim(ys.min() - 0.5, ys.max() + 0.5)
    fig.suptitle("Accuracy against deployment cost on Jetson Orin (TensorRT FP16, FASDD test split)", fontsize=10.5,
                 fontweight="bold", y=1.08)
    family_legend(fig, y=1.03)
    save(fig, "fig1_accuracy_vs_cost")


def fig_accuracy_by_domain(rows):
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    for ax, d in zip(axes, DOMAINS):
        scatter_points(ax, rows, lambda r: b1(r, "trt_fp16", "p50_batch_ms"), lambda r, d=d: 100 * r["acc"][d],
                       yerr=lambda r, d=d: (100 * r["ci"][d][0], 100 * r["ci"][d][1]), jitter=True, labels=False)
        lo = min(100 * r["ci"][d][0] for r in rows)
        hi = max(100 * r["ci"][d][1] for r in rows)
        ax.set_ylim(lo - 0.3 * (hi - lo) / 5 - 0.2, hi + 0.3 * (hi - lo) / 5 + 0.2)
        ax.set_xscale("log")
        ax.set_xlim(3.3, 32)
        ax.set_xticks([4, 6, 8, 12, 20])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
        n = rows[0]["n"][d]
        ax.set_title(f"{DOMAIN_LABEL[d]} (n = {n:,})", loc="left")
        ax.set_xlabel("Latency, batch 1, p50 (ms, log)")
        style_axes(ax)
    axes[0].set_ylabel("Accuracy (%), bars: 95% Wilson CI")
    fig.suptitle("Per-domain accuracy against latency (TensorRT FP16)", fontsize=10.5, fontweight="bold", y=1.14)
    family_legend(fig, y=1.08, shapes=True)
    fig.text(0.5, -0.075, "The three SwinV2-B variants have the same latency (11.7-11.9 ms); they are offset horizontally "
             "(-7% / 0 / +7.5%) so their intervals can be read.", ha="center", fontsize=7.5, color=MUTED)
    save(fig, "fig2_accuracy_by_domain_vs_latency")


def grouped_bars(ax, rows, key, xlabel, fmt):
    n_rt = len(RUNTIMES)
    height = 0.24
    ypos = np.arange(len(rows))[::-1]
    for j, (rt, label, color) in enumerate(RUNTIMES):
        vals = [b1(r, rt, key) for r in rows]
        y = ypos + (j - (n_rt - 1) / 2) * (height + 0.03)
        ax.barh(y, vals, height=height, color=color, label=label, zorder=3)
        for yy, v in zip(y, vals):
            ax.text(v * 1.08, yy, fmt(v), va="center", fontsize=7, color=INK2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r["name"] for r in rows])
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)


def fig_latency_energy_by_runtime(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9), sharey=True)
    grouped_bars(axes[0], rows, "p50_batch_ms", "Latency, batch 1, p50 (ms, log)", lambda v: f"{v:.1f}" if v < 100 else f"{v:.0f}")
    axes[0].set_title("Latency", loc="left")
    grouped_bars(axes[1], rows, "energy_mj_per_image", "Energy per image, batch 1 (mJ, log)", lambda v: f"{v:.0f}")
    axes[1].set_title("Energy per image", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for _, l, c in RUNTIMES]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=2.4)
    axes[0].set_xlim(1.5, 2500)
    axes[1].set_xlim(10, 50000)
    fig.suptitle("Runtime comparison on Jetson Orin, batch 1 (lower is better)", fontsize=10.5, fontweight="bold", y=1.09)
    save(fig, "fig3_latency_energy_by_runtime")


def fig_throughput_vs_batch(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.6), sharey=True)
    titles = {"ort_cuda": "ORT CUDA", "trt_fp16": "TensorRT FP16 (SwinV2 engines are batch-1 only)"}
    for ax, rt in zip(axes, ("ort_cuda", "trt_fp16")):
        swin_done = False
        for r in rows:
            batches = sorted(int(b) for b in r["bench"][rt])
            ys = [r["bench"][rt][str(b)]["throughput_images_per_second"] for b in batches]
            color = FAMILY_COLOR[r["family"]]
            label = None
            if r["family"] == "swin":
                if swin_done:
                    label = None
                else:
                    label, swin_done = "SwinV2-B (3 variants)", True
            else:
                label = r["name"]
            ax.plot(batches, ys, "-o", color=color, lw=2, markersize=6, markeredgecolor="white", markeredgewidth=1.2,
                    label=label, zorder=3)
            ax.annotate(f"{ys[-1]:.0f}", (batches[-1], ys[-1]), xytext=(7, 0), textcoords="offset points",
                        fontsize=7.5, color=INK2, va="center") if r["family"] != "swin" or r["model"] == "kd_routed_seed42" else None
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 8])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_yscale("log")
        ax.set_xlim(0.85, 11)
        ax.set_xlabel("Batch size")
        ax.set_title(titles[rt], loc="left")
        style_axes(ax)
        ax.legend(loc="upper left" if rt == "ort_cuda" else "lower right")
    axes[0].set_ylabel("Throughput (images/s, log)")
    fig.suptitle("Throughput against batch size on Jetson Orin (higher is better)", fontsize=10.5, fontweight="bold", y=1.04)
    save(fig, "fig4_throughput_vs_batch")


# ----------------------------------------------------------------------------- tables
def write_tables(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["model", "params_m", "macro_acc", "cv_acc", "uav_acc", "rs_acc", "trt_p50_ms_b1", "trt_img_s_b1", "trt_img_s_b8",
            "ortcuda_p50_ms_b1", "ortcpu_p50_ms_b1", "trt_mj_per_img_b1", "engine_mib"]
    table = []
    for r in rows:
        trt8 = r["bench"]["trt_fp16"].get("8")
        table.append({
            "model": r["model"], "params_m": r["params_m"], "macro_acc": round(r["macro_acc"], 4),
            "cv_acc": round(r["acc"]["cv"], 4), "uav_acc": round(r["acc"]["uav"], 4), "rs_acc": round(r["acc"]["rs"], 4),
            "trt_p50_ms_b1": round(b1(r, "trt_fp16", "p50_batch_ms"), 2),
            "trt_img_s_b1": round(b1(r, "trt_fp16", "throughput_images_per_second"), 1),
            "trt_img_s_b8": round(trt8["throughput_images_per_second"], 1) if trt8 else "",
            "ortcuda_p50_ms_b1": round(b1(r, "ort_cuda", "p50_batch_ms"), 2),
            "ortcpu_p50_ms_b1": round(b1(r, "ort_cpu", "p50_batch_ms"), 1),
            "trt_mj_per_img_b1": round(b1(r, "trt_fp16", "energy_mj_per_image"), 1), "engine_mib": round(r["engine_mib"], 1),
        })
    with (OUT / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    head = "| model | params (M) | macro acc | CV | UAV | RS | TRT p50 ms (b1) | TRT img/s (b1) | TRT img/s (b8) | ORT CUDA ms (b1) | ORT CPU ms (b1) | TRT mJ/img (b1) | engine MiB |"
    lines = [head, "|" + "---|" * 13]
    for t in table:
        lines.append("| " + " | ".join(str(t[c]) if c not in ("macro_acc", "cv_acc", "uav_acc", "rs_acc") else f"{100 * t[c]:.2f}" for c in cols) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    rows = collect()
    fig_accuracy_vs_cost(rows)
    fig_accuracy_by_domain(rows)
    fig_latency_energy_by_runtime(rows)
    fig_throughput_vs_batch(rows)
    write_tables(rows)
    print(f"Wrote figures and tables to {OUT}")


if __name__ == "__main__":
    main()
