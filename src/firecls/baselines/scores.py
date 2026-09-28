"""Turn per-class *presence* probabilities into the student's 4-way label space.

Detectors and segmenters do not produce 4-class logits; they tell us how confident they are
that fire is present and that smoke is present. Given calibrated thresholds ``t_fire`` and
``t_smoke`` the image-level decision is

    fire only  -> "fire",  smoke only -> "smoke",  both -> "both",  neither -> "neither".

To reuse the student's arg-max evaluation, ONNX export and TensorRT runners unchanged, each
presence probability ``p`` is mapped piece-wise linearly to ``q`` so that ``q > 0.5`` exactly
when ``p >= t``, and the four scores are the joint probabilities of two independent Bernoulli
variables with parameters ``q_fire`` and ``q_smoke``. The arg-max of those joint probabilities
is therefore identical to the thresholded decision, while the scores stay continuous.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from firecls.baselines.protocol import CLASSES
from firecls.evaluation import classification_metrics


# Gap kept around 0.5 so that p == t is not a tie between classes (q would be exactly 0.5), and
# so the decision survives FP16 rounding of q in TensorRT engines (FP16 spacing near 0.5 ~ 4.9e-4).
_MARGIN = 1e-3


def _rescale(p, t):
    """Piece-wise linear map [0,t)->[0,0.5-m), [t,1]->[0.5+m,1]. Works for numpy and torch."""
    lower = (0.5 - _MARGIN) * p / t
    upper = 0.5 + _MARGIN + (0.5 - _MARGIN) * (p - t) / (1.0 - t)
    if isinstance(p, torch.Tensor):
        return torch.where(p >= t, upper, lower)
    return np.where(p >= t, upper, lower)


def presence_to_scores(p_fire, p_smoke, t_fire: float = 0.5, t_smoke: float = 0.5, classes: Sequence[str] = CLASSES):
    """Return ``[batch, 4]`` scores ordered like ``classes``. Accepts numpy arrays or torch tensors."""
    _check_thresholds(t_fire, t_smoke)
    return _joint_scores(_rescale(p_fire, t_fire), _rescale(p_smoke, t_smoke), classes)


def _check_thresholds(t_fire: float, t_smoke: float) -> None:
    if not (0.0 < float(t_fire) < 1.0 and 0.0 < float(t_smoke) < 1.0):
        raise ValueError("Thresholds must lie strictly between 0 and 1.")


def _joint_scores(qf, qs, classes: Sequence[str] = CLASSES):
    joint = {
        "fire": qf * (1 - qs),
        "smoke": (1 - qf) * qs,
        "both": qf * qs,
        "neither": (1 - qf) * (1 - qs),
    }
    columns = [joint[name] for name in classes]
    if isinstance(qf, torch.Tensor):
        return torch.stack(columns, dim=1)
    return np.stack(columns, axis=1).astype(np.float32)


class PresenceToScores(torch.nn.Module):
    """Export-friendly head: ``(p_fire, p_smoke) -> [batch, 4]`` with thresholds baked in."""

    def __init__(self, t_fire: float = 0.5, t_smoke: float = 0.5) -> None:
        super().__init__()
        _check_thresholds(t_fire, t_smoke)
        # Plain floats are traced as constants, keeping the exported graph free of control flow.
        self.t_fire = float(t_fire)
        self.t_smoke = float(t_smoke)

    def forward(self, p_fire: torch.Tensor, p_smoke: torch.Tensor) -> torch.Tensor:
        return _joint_scores(_rescale(p_fire, self.t_fire), _rescale(p_smoke, self.t_smoke))


def calibrate_thresholds(
    domain_presence: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    grid: Sequence[float] | None = None,
    metric: str = "macro_f1_present_classes",
) -> dict:
    """Grid-search ``(t_fire, t_smoke)`` on the *validation* split.

    ``domain_presence`` maps domain -> ``(labels, p_fire, p_smoke)``. The objective is the
    equal-weight mean of ``metric`` over domains, i.e. the same aggregation used for reporting,
    so no domain (e.g. the small RS set) is swamped by CV.
    """
    grid = list(grid) if grid is not None else [round(v, 2) for v in np.arange(0.05, 0.96, 0.05)]
    best = {"t_fire": 0.5, "t_smoke": 0.5, "objective": -1.0}
    for t_fire in grid:
        for t_smoke in grid:
            values = []
            for labels, p_fire, p_smoke in domain_presence.values():
                predictions = presence_to_scores(p_fire, p_smoke, t_fire, t_smoke).argmax(axis=1)
                values.append(classification_metrics(labels, predictions, CLASSES)[metric])
            objective = float(np.mean(values))
            if objective > best["objective"]:
                best = {"t_fire": float(t_fire), "t_smoke": float(t_smoke), "objective": objective}
    best["metric"] = metric
    best["grid"] = grid
    return best
