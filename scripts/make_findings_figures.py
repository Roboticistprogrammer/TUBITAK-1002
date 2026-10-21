"""Numbers and figures for the thesis Findings chapter (all evaluated models, one consistent style).

    python scripts/make_findings_figures.py --teacher-preds <dir with teacher_{cv,uav,rs}/> \
        --histories <dir with {kd_routed,kd_uniform,nokd}/seed42/history.json> --out <thesis>/Figures/Findings

Accuracy of deployable models uses their TensorRT FP16 predictions on the Jetson; the teachers are
not deployed and use their PyTorch FP32 predictions. Writes ``findings_numbers.json`` (every value
quoted in the chapter) and three PDF figures. Baselines still training (ResNet-50, YOLO11n) are picked
up automatically once ``<arch>/predictions/<model>_trt_fp16`` exists, keeping their fixed colours.
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
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["fire", "smoke", "both", "neither"]
DOMAINS = ["cv", "uav", "rs"]
DLABEL = {"cv": "CV (ground)", "uav": "UAV", "rs": "RS (satellite)"}
# Fixed entity -> colour (validated categorical order, light surface). Never re-ordered.
MODELS = [  # key, label, predictions dir, benchmark prefix, colour
    ("kd_routed", "Student, KD routed", "swinv2-b-students/predictions/kd_routed_seed42", "swinv2-b-students/benchmarks/kd_routed_seed42", "#2a78d6"),
    ("kd_uniform", "Student, KD uniform", "swinv2-b-students/predictions/kd_uniform_seed42", "swinv2-b-students/benchmarks/kd_uniform_seed42", "#eb6834"),
    ("nokd", "Student, no KD", "swinv2-b-students/predictions/nokd_seed42", "swinv2-b-students/benchmarks/nokd_seed42", "#1baf7a"),
    ("mobilenet", "MobileNetV3-Large", "mobilenet_v3_large/predictions/mobilenet_v3_large_seed42", "mobilenet_v3_large/benchmarks/mobilenet_v3_large_seed42", "#eda100"),
    ("resnet50", "ResNet-50", "resnet50/predictions/resnet50_seed42", "resnet50/benchmarks/resnet50_seed42", "#e87ba4"),
    ("yolo11n", "YOLO11n", "yolo11n/predictions/yolo11n_coco_seed42", "yolo11n/benchmarks/yolo11n_coco_seed42", "#008300"),
]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                                               "#184f95", "#0d366b"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42, "savefig.facecolor": "white",
})
RNG_SEED, N_BOOT = 0, 2000


def load(d: Path, dom: str):
    """Predictions sorted by file name, so every model is compared on the same image order."""
    z = np.load(d / f"{dom}.npz", allow_pickle=False)
    order = np.argsort(z["names"], kind="stable")
    return z["logits"].argmax(1)[order], z["labels"][order], z["names"][order]


def f1s(y, p):
    out = {}
    for k, c in enumerate(CLASSES):
        tp, nt, npred = ((p == k) & (y == k)).sum(), (y == k).sum(), (p == k).sum()
        pr = tp / npred if npred else None
        rc = tp / nt if nt else None
        out[c] = {"support": int(nt), "precision": None if pr is None else float(pr),
                  "recall": None if rc is None else float(rc),
                  "f1": None if not nt else float(2 * pr * rc / (pr + rc)) if pr and rc else 0.0}
    return out


def metrics(preds, rng):
    res, boots = {}, []
    for d in DOMAINS:
        p, y, _ = preds[d]
        pc = f1s(y, p)
        present = [c for c in CLASSES if pc[c]["support"]]
        corr = (p == y).astype(float)
        b = rng.choice(corr, size=(N_BOOT, len(corr))).mean(1)
        boots.append(b)
        cm = np.zeros((4, 4), int)
        np.add.at(cm, (y, p), 1)
        fireish, neither = np.isin(y, [0, 2]), y == 3
        res[d] = {"n": int(len(y)), "acc": float(corr.mean()),
                  "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                  "macro_f1": float(np.mean([pc[c]["f1"] for c in present])),
                  "balanced_acc": float(np.mean([pc[c]["recall"] for c in present])),
                  "fire_recall": pc["fire"]["recall"],
                  "any_fire_recall": float(np.isin(p[fireish], [0, 2]).mean()),
                  "false_alarm": float((p[neither] != 3).mean()),
                  "per_class": pc, "cm": cm.tolist()}
    mb = np.mean(boots, 0)
    n = sum(res[d]["n"] for d in DOMAINS)
    res["macro"] = {"acc": float(np.mean([res[d]["acc"] for d in DOMAINS])),
                    "ci": [float(np.percentile(mb, 2.5)), float(np.percentile(mb, 97.5))],
                    "macro_f1": float(np.mean([res[d]["macro_f1"] for d in DOMAINS]))}
    res["weighted"] = {"acc": float(sum(res[d]["acc"] * res[d]["n"] for d in DOMAINS) / n)}
    return res


def mcnemar(a, b, d):
    (pa, y, na), (pb, yb, nb) = a[d], b[d]
    assert (na == nb).all() and (y == yb).all(), f"image order differs ({d})"
    ca, cb = pa == y, pb == y
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return {"delta_pp": float((ca.mean() - cb.mean()) * 100), "only_a": only_a, "only_b": only_b,
            "p": float(binomtest(only_a, only_a + only_b, 0.5).pvalue) if only_a + only_b else 1.0}


def bench(prefix: Path, rt: str):
    f = prefix.parent / f"{prefix.name}_{rt}.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text())
    r = next(iter(j["runtimes"].values()))
    b = r["batches"]["1"]
    idle = j["environment"].get("idle_ram_used_mb")
    return {"p50": b["p50_batch_ms"], "p95": b["p95_batch_ms"], "p99": b.get("p99_batch_ms"),
            "fps": b["throughput_images_per_second"], "power_w": b["power"]["mean_mw"] / 1000,
            "energy_mj": b["energy_mj_per_image"],
            "ram_above_idle_mb": b["memory"]["peak_ram_mb"] - idle if idle else None,
            "artifact_mib": r.get("artifact_mib"), "batches": sorted(int(k) for k in r["batches"])}


def bench_all(prefix: Path, rt: str):
    f = prefix.parent / f"{prefix.name}_{rt}.json"
    if not f.exists():
        return None
    r = next(iter(json.loads(f.read_text())["runtimes"].values()))
    return {int(k): {"p50": v["p50_batch_ms"], "fps": v["throughput_images_per_second"]} for k, v in r["batches"].items()}


def holm(pvals: dict) -> dict:
    """Holm step-down adjustment over all keys of ``pvals``."""
    order = sorted(pvals, key=pvals.get)
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def fig_cross_domain(acc, out):
    fig, ax = plt.subplots(figsize=(4.6, 3.4), constrained_layout=True)
    m = np.array([[acc[t][d] * 100 for d in DOMAINS] for t in DOMAINS])
    ax.imshow(m, cmap=BLUES, vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if m[i, j] > 55 else INK)
    ax.set_xticks(range(3), [DLABEL[d] for d in DOMAINS])
    ax.set_yticks(range(3), [f"{d.upper()} teacher" for d in DOMAINS])
    ax.set_xlabel("Test domain")
    ax.tick_params(length=0, which="both")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, 3), minor=True)
    ax.set_yticks(np.arange(-.5, 3), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    save(fig, out, "teacher_cross_domain")


def save(fig, out, stem):
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_accuracy(res, teacher, models, out):
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 0.55 * len(models) + 1.4), sharey=True, constrained_layout=True)
    ys = np.arange(len(models))[::-1]
    for ax, key, title in zip(axes, DOMAINS + ["macro"], [DLABEL[d] for d in DOMAINS] + ["Macro average"]):
        lo = [res[m[0]][key]["ci"][0] * 100 for m in models]
        hi = [res[m[0]][key]["ci"][1] * 100 for m in models]
        t = teacher[key]["acc"] * 100
        span = max(max(hi), t) - min(min(lo), t)
        for y, m, a, b in zip(ys, models, lo, hi):
            v = res[m[0]][key]["acc"] * 100
            ax.plot([a, b], [y, y], color=m[4], lw=2, solid_capstyle="round", zorder=2)
            ax.plot(v, y, "o", ms=7, color=m[4], mec="white", mew=1.5, zorder=3)
            ax.text(b + 0.05 * span, y, f"{v:.1f}", va="center", fontsize=7.5, color=INK, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
        ax.axvline(t, color=MUTED, lw=1.2, ls=(0, (3, 2)), zorder=1)
        ax.text(t, len(models) - 0.45, f"teacher {t:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xlim(min(min(lo), t) - 0.12 * span, max(max(hi), t) + 0.4 * span)
        ax.set_ylim(-0.6, len(models) + 0.1)
        ax.set_yticks(ys, [m[1] for m in models])
        ax.xaxis.grid(True, zorder=0)
        ax.set_title(title, pad=14)
        ax.set_xlabel("Accuracy (%)")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    save(fig, out, "accuracy_ci")


def fig_confusion(r, out, stem):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), constrained_layout=True)
    for ax, d in zip(axes, DOMAINS):
        cm = np.array(r[d]["cm"], float)
        rows = cm.sum(1, keepdims=True)
        pct = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0) * 100
        ax.imshow(pct, cmap=BLUES, vmin=0, vmax=100)
        for i in range(4):
            if rows[i, 0] == 0:
                ax.add_patch(plt.Rectangle((-0.5, i - 0.5), 4, 1, facecolor="#f0efec", edgecolor="none"))
                ax.text(1.5, i, "no test images", ha="center", va="center", color=MUTED, fontsize=7.5, style="italic")
                continue
            for j in range(4):
                ax.text(j, i, f"{pct[i, j]:.1f}%\n({int(cm[i, j])})", ha="center", va="center", fontsize=6.8,
                        color="white" if pct[i, j] > 55 else INK, linespacing=1.15)
        ax.set_xticks(range(4), CLASSES)
        ax.set_yticks(range(4), [f"{c} ({int(n)})" for c, n in zip(CLASSES, rows[:, 0])])
        ax.set_xlabel("Predicted class")
        if d == "cv":
            ax.set_ylabel("True class (test images)")
        ax.set_title(f"{DLABEL[d]}  ·  accuracy {r[d]['acc'] * 100:.1f}%")
        ax.tick_params(length=0, which="both")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, 4), minor=True)
        ax.set_yticks(np.arange(-.5, 4), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
    save(fig, out, stem)


def fig_training(hist, out):
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 2.8), sharex=True, constrained_layout=True)
    for ax, key, title in zip(axes, ["avg"] + DOMAINS, ["Mean over domains"] + [DLABEL[d] for d in DOMAINS]):
        for m, recs in hist:
            ep = [r["epoch"] for r in recs]
            v = [(r["avg_val"] if key == "avg" else r["val_acc"][key]) * 100 for r in recs]
            ax.plot(ep, v, color=m[4], lw=2, label=m[1], zorder=3)
            if key == "avg":
                best = max(recs, key=lambda r: r["avg_val"])
                ax.plot(best["epoch"], best["avg_val"] * 100, "o", ms=7, color=m[4], mec="white", mew=1.5, zorder=4)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.yaxis.grid(True, zorder=0)
        ax.set_xticks([1, 5, 10, 15, 20])
    axes[0].set_ylabel("Validation accuracy (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", ncols=3, bbox_to_anchor=(0.0, 1.1))
    save(fig, out, "training_curves")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher-preds", type=Path, required=True)
    ap.add_argument("--histories", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    teacher_preds = {d: load(a.teacher_preds / f"teacher_{d}", d) for d in DOMAINS}
    teacher = metrics(teacher_preds, rng)
    models = [m for m in MODELS if (ROOT / f"{m[2]}_trt_fp16").exists()]
    preds, onnx, res, res_onnx = {}, {}, {}, {}
    for m in models:
        preds[m[0]] = {d: load(ROOT / f"{m[2]}_trt_fp16", d) for d in DOMAINS}
        onnx[m[0]] = {d: load(ROOT / f"{m[2]}_onnx", d) for d in DOMAINS}
        res[m[0]] = metrics(preds[m[0]], rng)
        res_onnx[m[0]] = metrics(onnx[m[0]], rng)

    pairs = [("kd_routed", "teacher"), ("kd_routed", "kd_uniform"), ("kd_routed", "nokd"),
             ("kd_uniform", "nokd"), ("kd_routed", "mobilenet"), ("kd_routed", "resnet50")]
    allp = {**preds, "teacher": teacher_preds}
    mcn = {f"{x} vs {y}": {d: mcnemar(allp[x], allp[y], d) for d in DOMAINS} for x, y in pairs if x in allp and y in allp}
    fp16 = {m[0]: {d: {"changed": int((preds[m[0]][d][0] != onnx[m[0]][d][0]).sum()),
                       "delta_pp": (res[m[0]][d]["acc"] - res_onnx[m[0]][d]["acc"]) * 100} for d in DOMAINS}
            for m in models}
    dep = {m[0]: {rt: bench(ROOT / m[3], rt) for rt in ("ort_cpu", "ort_cuda", "trt_fp16")} for m in models}
    dep_all = {m[0]: {rt: bench_all(ROOT / m[3], rt) for rt in ("ort_cpu", "ort_cuda", "trt_fp16")} for m in models}
    flat = {f"{k}|{d}": v[d]["p"] for k, v in mcn.items() for d in DOMAINS}
    for key, p_adj in holm(flat).items():
        k, d = key.split("|")
        mcn[k][d]["p_holm"] = p_adj
    teacher_dirs = {t: a.teacher_preds / f"teacher_{t}" for t in DOMAINS}
    cross = {t: {d: float((lambda pr: (pr[0] == pr[1]).mean())(load(teacher_dirs[t], d))) for d in DOMAINS}
             for t in DOMAINS}
    teacher_own = {}
    for t in DOMAINS:
        teacher_own[t] = metrics({d: load(teacher_dirs[t], d) for d in DOMAINS}, rng)[t]
    hist = []
    for m in models[:3]:
        h = json.loads((a.histories / m[0] / "seed42" / "history.json").read_text())["records"]
        hist.append((m, h))

    strip = lambda r: {k: v for k, v in r.items()}
    numbers = {"teacher": teacher, "models": {m[0]: {"label": m[1], "trt_fp16": res[m[0]], "onnx": res_onnx[m[0]]}
                                              for m in models},
               "retention": {m[0]: res[m[0]]["macro"]["acc"] / teacher["macro"]["acc"] for m in models},
               "mcnemar": mcn, "n_mcnemar_tests": len(flat), "fp16_vs_onnx": fp16, "deployment_batch1": dep,
               "deployment_all_batches": dep_all, "teacher_cross_domain": cross,
               "teacher_own_domain": teacher_own,
               "best_epoch": {m[0]: max(h, key=lambda r: r["avg_val"])["epoch"] for m, h in hist},
               "best_val": {m[0]: max(r["avg_val"] for r in h) for m, h in hist}}
    (a.out / "findings_numbers.json").write_text(json.dumps(numbers, indent=1, default=strip) + "\n")
    fig_accuracy(res, teacher, models, a.out)
    fig_confusion(res["kd_routed"], a.out, "confusion_kd_routed")
    for k in ("kd_uniform", "nokd", "mobilenet", "resnet50"):
        if k in res:
            fig_confusion(res[k], a.out, f"confusion_{k}")
    fig_confusion(teacher_own, a.out, "confusion_teachers")
    fig_cross_domain(cross, a.out)
    fig_training(hist, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
