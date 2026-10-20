"""Thesis figures and metrics for one baseline's test-dataset results (default: MobileNetV3-Large).

    python scripts/plot_baseline_results.py
    python scripts/plot_baseline_results.py --arch-dir resnet50 --model resnet50_seed42 --label ResNet-50

Reads the per-image logits in ``<arch-dir>/predictions/<model>_{onnx,trt_fp16}/{cv,uav,rs}.npz`` and
the Jetson benchmarks in ``<arch-dir>/benchmarks/``, and compares against the SwinV2-B students in
``swinv2-b-students/``. Accuracy figures use the TensorRT FP16 predictions (the deployed engine).
Writes PDF (vector, for LaTeX) and PNG (300 dpi) figures plus ``metrics.json`` / ``metrics.md``
into ``<arch-dir>/figures/``.

Metric definitions follow thesis Section 3.3.3: macro-F1 is taken over the classes present in the
test set of each domain (summary.csv averages over all four classes, which includes an absent,
zero-score class in RS), any-fire recall counts fire/both images predicted as fire or both, and the
false-alarm rate is the share of neither images predicted as fire, smoke or both.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["fire", "smoke", "both", "neither"]
DOMAINS = ["cv", "uav", "rs"]
DOMAIN_LABEL = {"cv": "CV (ground)", "uav": "UAV", "rs": "RS (satellite)"}
STUDENTS = [("kd_routed_seed42", "SwinV2-B, KD routed"),
            ("kd_uniform_seed42", "SwinV2-B, KD uniform"),
            ("nokd_seed42", "SwinV2-B, no KD")]
RUNTIMES = [("ort_cpu", "ORT CPU"), ("ort_cuda", "ORT CUDA"), ("trt_fp16", "TensorRT FP16")]

# Validated categorical palette (light surface), fixed order; slot 1 is always the baseline.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                    "#256abf", "#184f95", "#0d366b"])
N_BOOT = 2000

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.labelsize": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "legend.fontsize": 8, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
})


# ----------------------------------------------------------------------------- data
def load(pred_dir: Path, domain: str):
    z = np.load(pred_dir / f"{domain}.npz", allow_pickle=False)
    return z["logits"].argmax(1), z["labels"], z["names"]


def per_class(y, p):
    out = {}
    for k, c in enumerate(CLASSES):
        tp, n_true, n_pred = int(((p == k) & (y == k)).sum()), int((y == k).sum()), int((p == k).sum())
        prec = tp / n_pred if n_pred else None
        rec = tp / n_true if n_true else None
        f1 = (2 * prec * rec / (prec + rec) if prec and rec else 0.0) if n_true else None
        out[c] = {"support": n_true, "predicted": n_pred, "precision": prec, "recall": rec, "f1": f1}
    return out


def domain_metrics(y, p, rng):
    pc = per_class(y, p)
    present = [c for c in CLASSES if pc[c]["support"]]
    f1_all4 = [pc[c]["f1"] if pc[c]["f1"] is not None else 0.0 for c in CLASSES]
    correct = (y == p).astype(float)
    boot = rng.choice(correct, size=(N_BOOT, len(correct)), replace=True).mean(1)
    fireish = np.isin(y, [0, 2])
    neither = y == 3
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (y, p), 1)
    return {
        "n": int(len(y)),
        "accuracy": float(correct.mean()),
        "accuracy_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "balanced_accuracy": float(np.mean([pc[c]["recall"] for c in present])),
        "macro_f1_present": float(np.mean([pc[c]["f1"] for c in present])),
        "macro_f1_all4": float(np.mean(f1_all4)),
        "fire_recall": pc["fire"]["recall"],
        "any_fire_recall": float(np.isin(p[fireish], [0, 2]).mean()) if fireish.any() else None,
        "false_alarm_rate": float((p[neither] != 3).mean()) if neither.any() else None,
        "per_class": pc,
        "confusion_matrix": cm.tolist(),
        "_boot": boot,
    }


def evaluate(pred_dir: Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    res, preds = {}, {}
    for d in DOMAINS:
        p, y, names = load(pred_dir, d)
        res[d] = domain_metrics(y, p, rng)
        preds[d] = (p, y, names)
    boots = np.stack([res[d]["_boot"] for d in DOMAINS])
    n = np.array([res[d]["n"] for d in DOMAINS])
    macro_boot = boots.mean(0)
    res["macro_avg"] = {
        "accuracy": float(np.mean([res[d]["accuracy"] for d in DOMAINS])),
        "accuracy_ci95": [float(np.percentile(macro_boot, 2.5)), float(np.percentile(macro_boot, 97.5))],
        "macro_f1_present": float(np.mean([res[d]["macro_f1_present"] for d in DOMAINS])),
    }
    res["weighted_avg"] = {"accuracy": float(sum(res[d]["accuracy"] * res[d]["n"] for d in DOMAINS) / n.sum())}
    return res, preds


def mcnemar(a_correct, b_correct):
    b = int((a_correct & ~b_correct).sum())
    c = int((~a_correct & b_correct).sum())
    pval = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"only_first_correct": b, "only_second_correct": c, "p_value": float(pval)}


def bench(path: Path):
    j = json.loads(path.read_text())
    rt = next(iter(j["runtimes"].values()))
    return {int(k): v for k, v in rt["batches"].items()}, rt.get("artifact_mib")


# ----------------------------------------------------------------------------- figures
def save(fig, out: Path, stem: str):
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_confusion(res, label, out):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), constrained_layout=True)
    for ax, d in zip(axes, DOMAINS):
        cm = np.array(res[d]["confusion_matrix"], float)
        rows = cm.sum(1, keepdims=True)
        pct = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0) * 100
        ax.imshow(pct, cmap=BLUES, vmin=0, vmax=100)
        for i in range(4):
            if rows[i, 0] == 0:
                ax.add_patch(plt.Rectangle((-0.5, i - 0.5), 4, 1, facecolor="#f0efec", edgecolor="none"))
                ax.text(1.5, i, "no test images", ha="center", va="center", color=MUTED, fontsize=7.5,
                        style="italic")
                continue
            for j in range(4):
                v = pct[i, j]
                ax.text(j, i, f"{v:.1f}%\n({int(cm[i, j])})", ha="center", va="center", fontsize=6.8,
                        color="white" if v > 55 else INK, linespacing=1.15)
        ax.set_xticks(range(4), CLASSES)
        ax.set_yticks(range(4), [f"{c} ({int(r)})" for c, r in zip(CLASSES, rows[:, 0])])
        ax.set_xlabel("Predicted class")
        if d == "cv":
            ax.set_ylabel("True class (test images)")
        ax.set_title(f"{DOMAIN_LABEL[d]}  ·  acc {res[d]['accuracy'] * 100:.1f}%")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, 4), minor=True)
        ax.set_yticks(np.arange(-.5, 4), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
    fig.suptitle(f"{label} (TensorRT FP16): confusion matrices on the FASDD test split, row-normalised",
                 fontsize=10, fontweight="bold", color=INK)
    save(fig, out, "fig1_confusion_matrices")


def fig_per_class(res, label, out):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=True, constrained_layout=True)
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    w = 0.26
    for ax, d in zip(axes, DOMAINS):
        pc = res[d]["per_class"]
        x = np.arange(4)
        for m, (key, name) in enumerate(metrics):
            vals = [pc[c][key] if pc[c]["support"] else np.nan for c in CLASSES]
            ax.bar(x + (m - 1) * w, vals, w - 0.03, color=SERIES[m], label=name, zorder=2)
            if key == "f1":
                for xi, v in zip(x, vals):
                    if not np.isnan(v):
                        ax.text(xi + w, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.6,
                                color=INK2)
        for xi, c in zip(x, CLASSES):
            if not pc[c]["support"]:
                ax.text(xi, 0.05, "no test\nimages", ha="center", va="bottom", fontsize=7, color=MUTED,
                        style="italic")
        ax.set_xticks(x, [f"{c}\n(n={pc[c]['support']})" for c in CLASSES])
        ax.set_ylim(0, 1.08)
        ax.set_xlim(-0.55, 3.55)
        ax.yaxis.grid(True, zorder=0)
        ax.set_title(DOMAIN_LABEL[d])
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Score")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", ncols=3, bbox_to_anchor=(0.0, 1.11),
               handlelength=1.2)
    fig.suptitle(f"{label} (TensorRT FP16): per-class precision, recall and F1 (F1 labelled)",
                 fontsize=10, fontweight="bold", color=INK, y=1.18)
    save(fig, out, "fig2_per_class_metrics")


def fig_accuracy_vs_students(all_res, out):
    panels = DOMAINS + ["macro_avg"]
    titles = [DOMAIN_LABEL[d] for d in DOMAINS] + ["Macro average"]
    names = list(all_res)
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.6), sharey=True, constrained_layout=True)
    for ax, key, title in zip(axes, panels, titles):
        vals = [all_res[n][key]["accuracy"] * 100 for n in names]
        lo = [all_res[n][key]["accuracy_ci95"][0] * 100 for n in names]
        hi = [all_res[n][key]["accuracy_ci95"][1] * 100 for n in names]
        ys = np.arange(len(names))[::-1]
        for k, (y, v, a, b) in enumerate(zip(ys, vals, lo, hi)):
            ax.plot([a, b], [y, y], color=SERIES[k], lw=2, solid_capstyle="round", zorder=2)
            ax.plot(v, y, "o", ms=7, color=SERIES[k], mec="white", mew=1.5, zorder=3)
        span = max(hi) - min(lo)
        ax.set_xlim(min(lo) - 0.15 * span, max(hi) + 0.45 * span)
        for k, (y, v, b) in enumerate(zip(ys, vals, hi)):
            ax.text(b + 0.06 * span, y, f"{v:.1f}", va="center", fontsize=7.5, color=INK)
        ax.set_yticks(ys, names)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.xaxis.grid(True, zorder=0)
        ax.set_title(title)
        ax.set_xlabel("Accuracy (%)")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    fig.suptitle("Test accuracy with bootstrap 95% CI (TensorRT FP16, seed 42; RS n = 370)",
                 fontsize=10, fontweight="bold", color=INK)
    save(fig, out, "fig3_accuracy_vs_students")


def fig_runtime(bm, label, out):
    batches = sorted(next(iter(bm.values())).keys())
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), constrained_layout=True)
    specs = [
        ("Latency per image, p50 (ms)", lambda b, s: s["p50_batch_ms"] / b, lambda b, s: s["p95_batch_ms"] / b),
        ("Throughput (images/s)", lambda b, s: s["throughput_images_per_second"], None),
        ("Energy per image (mJ)", lambda b, s: s["energy_mj_per_image"], None),
    ]
    for ax, (title, f, upper) in zip(axes, specs):
        for k, (rt, name) in enumerate(RUNTIMES):
            ys = [f(b, bm[rt][b]) for b in batches]
            ax.plot(batches, ys, "-o", color=SERIES[k], lw=2, ms=6, mec="white", mew=1.2, label=name, zorder=3)
            if upper:
                for b, y in zip(batches, ys):
                    ax.plot([b, b], [y, upper(b, bm[rt][b])], color=SERIES[k], lw=1, alpha=0.7, zorder=2)
            ax.annotate(f"{ys[-1]:.0f}" if ys[-1] >= 10 else f"{ys[-1]:.1f}", (batches[-1], ys[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(batches)
        ax.set_xlim(batches[0] - 0.6, batches[-1] + 1.6)
        ax.set_xlabel("Batch size")
        ax.set_title(title)
        ax.yaxis.grid(True, which="major", zorder=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", ncols=3, bbox_to_anchor=(0.0, 1.11))
    axes[0].text(0.02, 0.02, "whisker: p95", transform=axes[0].transAxes, fontsize=7, color=MUTED)
    fig.suptitle(f"{label} on Jetson Orin: runtime and batch size (192 × 192 input, preprocessing excluded)",
                 fontsize=10, fontweight="bold", color=INK, y=1.18)
    save(fig, out, "fig4_runtime_latency_throughput_energy")


def fig_deployment(dep, out):
    names = list(dep)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.4), sharey=True, constrained_layout=True)
    specs = [("p50 latency, batch 1 (ms)", "latency"), ("Energy per image (mJ)", "energy"),
             ("TensorRT engine size (MiB)", "size")]
    ys = np.arange(len(names))[::-1]
    for ax, (title, key) in zip(axes, specs):
        vals = [dep[n][key] for n in names]
        ax.barh(ys, vals, 0.62, color=SERIES[:len(names)], zorder=2)
        for y, v in zip(ys, vals):
            ax.text(v + max(vals) * 0.02, y, f"{v:.1f}", va="center", fontsize=7.5, color=INK)
        ax.set_xlim(0, max(vals) * 1.22)
        ax.set_yticks(ys, names)
        ax.set_title(title)
        ax.xaxis.grid(True, zorder=0)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Deployment cost on Jetson Orin, TensorRT FP16 (student engines built for batch 1 only)",
                 fontsize=10, fontweight="bold", color=INK)
    save(fig, out, "fig5_deployment_vs_students")


# ----------------------------------------------------------------------------- report
def pct(v):
    return "–" if v is None else f"{v * 100:.1f}"


def write_report(label, model, res, onnx_diff, mcn, dep, filename_check, out):
    clean = {k: {kk: vv for kk, vv in v.items() if kk != "_boot"} if isinstance(v, dict) else v
             for k, v in res.items()}
    (out / "metrics.json").write_text(json.dumps({
        "model": model, "runtime": "TensorRT FP16", "domains": clean, "onnx_vs_trt_fp16_disagreements": onnx_diff,
        "mcnemar_vs_students": mcn, "deployment_batch1_trt_fp16": dep, "filename_prefix_check": filename_check,
    }, indent=2) + "\n")
    L = [f"# {label} ({model}): test-split metrics and figures", "",
         "TensorRT FP16 engine predictions on the official FASDD test split. Generated by "
         "`scripts/plot_baseline_results.py`. Seed 42 only, so no seed variance yet.", "",
         "## Per-domain metrics (%)", "",
         "| Domain | n | Accuracy [95% CI] | Balanced acc. | Macro-F1 (present classes) | Macro-F1 (all 4) "
         "| Fire recall | Any-fire recall | False-alarm rate |",
         "|---|---|---|---|---|---|---|---|---|"]
    for d in DOMAINS:
        r = res[d]
        L.append(f"| {DOMAIN_LABEL[d]} | {r['n']} | {pct(r['accuracy'])} [{pct(r['accuracy_ci95'][0])}, "
                 f"{pct(r['accuracy_ci95'][1])}] | {pct(r['balanced_accuracy'])} | {pct(r['macro_f1_present'])} "
                 f"| {pct(r['macro_f1_all4'])} | {pct(r['fire_recall'])} | {pct(r['any_fire_recall'])} "
                 f"| {pct(r['false_alarm_rate'])} |")
    m = res["macro_avg"]
    L += [f"| **Macro average** | | {pct(m['accuracy'])} [{pct(m['accuracy_ci95'][0])}, "
          f"{pct(m['accuracy_ci95'][1])}] | | {pct(m['macro_f1_present'])} | | | | |",
          f"| Sample-weighted | 20435 | {pct(res['weighted_avg']['accuracy'])} | | | | | | |", "",
          "Macro-F1 (all 4) matches `summary.csv`; in RS it averages in the absent `fire` class at 0, which is "
          "why it is much lower. The thesis definition (Section 3.3.3) is the present-classes column.", "",
          "## Per-class precision / recall / F1 (%)", "",
          "| Domain | Class | Support | Precision | Recall | F1 |", "|---|---|---|---|---|---|"]
    for d in DOMAINS:
        for c in CLASSES:
            pc = res[d]["per_class"][c]
            L.append(f"| {DOMAIN_LABEL[d]} | {c} | {pc['support']} | {pct(pc['precision'])} | {pct(pc['recall'])} "
                     f"| {pct(pc['f1'])} |")
    L += ["", "## ONNX (CPU) vs TensorRT FP16", "",
          "Images whose predicted class differs between the two runtimes: " +
          ", ".join(f"{DOMAIN_LABEL[d]} {onnx_diff[d]}" for d in DOMAINS) + ".", "",
          "## Exact McNemar test against the SwinV2-B students (TensorRT FP16, same test images)", "",
          f"Column b = images only {label} classifies correctly; c = images only the student classifies correctly.",
          "", "| Student | Domain | b | c | p |", "|---|---|---|---|---|"]
    for s, rows in mcn.items():
        for d in DOMAINS:
            r = rows[d]
            L.append(f"| {s} | {DOMAIN_LABEL[d]} | {r['only_first_correct']} | {r['only_second_correct']} "
                     f"| {r['p_value']:.3g} |")
    L += ["", "## Deployment at batch 1 (TensorRT FP16, Jetson Orin)", "",
          "| Model | p50 latency (ms) | Energy per image (mJ) | Engine size (MiB) |", "|---|---|---|---|"]
    for n, v in dep.items():
        L.append(f"| {n} | {v['latency']:.1f} | {v['energy']:.1f} | {v['size']:.1f} |")
    L += ["", "## Label check: presence-rule label vs FASDD filename prefix", "",
          "| Domain | Images | Agree | Disagree | Disagreements |", "|---|---|---|---|---|"]
    for d in DOMAINS:
        f = filename_check[d]
        detail = "; ".join(f"{k}: {v}" for k, v in f["pairs"].items()) or "–"
        L.append(f"| {DOMAIN_LABEL[d]} | {f['n']} | {f['agree']} | {f['n'] - f['agree']} | {detail} |")
    L += ["", "## Figures", "",
          "| File | Content |", "|---|---|",
          "| `fig1_confusion_matrices` | Row-normalised confusion matrix per domain, with counts |",
          "| `fig2_per_class_metrics` | Per-class precision, recall and F1 per domain |",
          "| `fig3_accuracy_vs_students` | Accuracy with bootstrap 95% CI per domain and macro average, against the "
          "three SwinV2-B students |",
          "| `fig4_runtime_latency_throughput_energy` | Latency per image, throughput and energy per image vs batch "
          "size for ORT CPU, ORT CUDA and TensorRT FP16 |",
          "| `fig5_deployment_vs_students` | Batch-1 latency, energy per image and engine size against the students |",
          "", "Each figure is saved as PDF (vector, for LaTeX) and PNG (300 dpi).", ""]
    (out / "metrics.md").write_text("\n".join(L))


PREFIX_TO_CLASS = {"fire": "fire", "smoke": "smoke", "bothFireAndSmoke": "both", "neitherFireNorSmoke": "neither"}


def filename_prefix_check(preds):
    out = {}
    for d in DOMAINS:
        _, y, names = preds[d]
        pairs, agree = {}, 0
        for n, l in zip(names, y):
            pref = PREFIX_TO_CLASS.get(str(n).split("_")[0], str(n).split("_")[0])
            if pref == CLASSES[l]:
                agree += 1
            else:
                key = f"prefix {pref} → label {CLASSES[l]}"
                pairs[key] = pairs.get(key, 0) + 1
        out[d] = {"n": int(len(y)), "agree": agree, "pairs": pairs}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch-dir", default="mobilenet_v3_large")
    ap.add_argument("--model", default="mobilenet_v3_large_seed42")
    ap.add_argument("--label", default="MobileNetV3-Large")
    ap.add_argument("--students-dir", default="swinv2-b-students")
    args = ap.parse_args()

    arch, students = ROOT / args.arch_dir, ROOT / args.students_dir
    out = arch / "figures"
    out.mkdir(exist_ok=True)

    res, preds = evaluate(arch / "predictions" / f"{args.model}_trt_fp16")
    _, onnx_preds = evaluate(arch / "predictions" / f"{args.model}_onnx")
    onnx_diff = {d: int((preds[d][0] != onnx_preds[d][0]).sum()) for d in DOMAINS}

    all_res, mcn = {args.label: res}, {}
    for key, name in STUDENTS:
        s_res, s_preds = evaluate(students / "predictions" / f"{key}_trt_fp16")
        all_res[name] = s_res
        mcn[name] = {}
        for d in DOMAINS:
            assert (s_preds[d][2] == preds[d][2]).all(), f"image order differs for {key}/{d}"
            mcn[name][d] = mcnemar(preds[d][0] == preds[d][1], s_preds[d][0] == s_preds[d][1])

    bm = {rt: bench(arch / "benchmarks" / f"{args.model}_{rt}.json")[0] for rt, _ in RUNTIMES}
    dep = {}
    for name, path in [(args.label, arch / "benchmarks" / f"{args.model}_trt_fp16.json")] + \
                      [(n, students / "benchmarks" / f"{k}_trt_fp16.json") for k, n in STUDENTS]:
        b, mib = bench(path)
        dep[name] = {"latency": b[1]["p50_batch_ms"], "energy": b[1]["energy_mj_per_image"], "size": mib}

    fig_confusion(res, args.label, out)
    fig_per_class(res, args.label, out)
    fig_accuracy_vs_students(all_res, out)
    fig_runtime(bm, args.label, out)
    fig_deployment(dep, out)
    write_report(args.label, args.model, res, onnx_diff, mcn, dep, filename_prefix_check(preds), out)
    print(f"Wrote figures and metrics to {out}")


if __name__ == "__main__":
    main()
