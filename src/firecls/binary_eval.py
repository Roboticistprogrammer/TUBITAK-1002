"""Binary smoke/fire-vs-neither evaluation of the four-class models (external/PLUME_PROTOCOL.md, Sections 1 and 8).

Ground truth positive: FASDD-RS `smoke` or `both` (fire never occurs alone in FASDD-RS), or a plume image; negative: `neither`
or a non-plume image. A prediction is positive if the predicted class (argmax of the four logits) is `fire`, `smoke` or `both`,
and negative if it is `neither`. The threshold-free score is 1 - P(neither) from the softmax of the logits.
"""
from __future__ import annotations

from math import comb
from typing import Sequence

import numpy as np

CLASSES = ["fire", "smoke", "both", "neither"]
FIRE, SMOKE, BOTH, NEITHER = range(4)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64) - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def positive_prediction(logits: np.ndarray) -> np.ndarray:
    """True where the predicted class is fire, smoke or both."""
    return logits.argmax(1) != NEITHER


def positive_score(logits: np.ndarray) -> np.ndarray:
    """1 - P(neither)."""
    return 1.0 - softmax(logits)[:, NEITHER]


def positive_truth(labels4: np.ndarray) -> np.ndarray:
    """True where the four-class ground truth is fire, smoke or both."""
    return np.asarray(labels4) != NEITHER


def roc_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based ROC-AUC with average ranks for ties."""
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_metrics(pred_pos: np.ndarray, score: np.ndarray, truth_pos: np.ndarray) -> dict:
    tp = int((pred_pos & truth_pos).sum())
    fn = int((~pred_pos & truth_pos).sum())
    fp = int((pred_pos & ~truth_pos).sum())
    tn = int((~pred_pos & ~truth_pos).sum())
    recall = tp / (tp + fn) if tp + fn else float("nan")
    far = fp / (fp + tn) if fp + tn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn, "errors": fn + fp,
        "recall": recall, "false_alarm_rate": far, "balanced_accuracy": (recall + 1 - far) / 2,
        "f1": f1, "roc_auc": roc_auc(score, truth_pos),
    }


def recall_on(pred_pos: np.ndarray, labels4: np.ndarray, label: int) -> dict:
    """Binary recall restricted to the images of one four-class label (e.g. the 31 `smoke` or the 191 `both` images)."""
    mask = np.asarray(labels4) == label
    return {"images": int(mask.sum()), "recall": float(pred_pos[mask].mean()) if mask.any() else float("nan")}


def bootstrap_ci(pred_pos: np.ndarray, score: np.ndarray, truth_pos: np.ndarray, resamples: int = 2000,
                 seed: int = 0) -> dict:
    """95% percentile bootstrap over images for recall, false-alarm rate, balanced accuracy, F1 and ROC-AUC."""
    rng = np.random.default_rng(seed)
    n = len(pred_pos)
    keys = ("recall", "false_alarm_rate", "balanced_accuracy", "f1", "roc_auc")
    draws = {k: [] for k in keys}
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        metrics = binary_metrics(pred_pos[idx], score[idx], truth_pos[idx])
        for k in keys:
            draws[k].append(metrics[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items()}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def holm(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def paired_test(pred_a: np.ndarray, pred_b: np.ndarray, truth_pos: np.ndarray) -> dict:
    """McNemar on binary correctness of two models over the same images. b: only A right, c: only B right."""
    ok_a, ok_b = pred_a == truth_pos, pred_b == truth_pos
    b, c = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_unadjusted": mcnemar_exact(b, c)}
