"""Side-by-side validation numbers of fine-tuning runs (e.g. the treatment RS teacher next to the control RS teacher).

    python -I scripts/plumes/compare_runs.py <out-root>/teacher_rs/seed42 <out-root>/teacher_rs_noplume/seed42 [...]

For every run directory (history.json + summary.json): the validation record of epoch 0 (the before model) and of the selected
epoch: CV and UAV 4-class accuracy (if the model has them), binary recall / false-alarm rate / balanced accuracy / ROC-AUC on
FASDD-RS val and on plume val, the pooled RS binary balanced accuracy, `both` recall, and the selection score. These are
validation numbers (the sets that pick the checkpoint), not test results. Standard library only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    runs = [Path(p) for p in sys.argv[1:]]
    if not runs:
        sys.exit(__doc__)
    cols = []
    for i, run in enumerate(runs):
        history, summary = json.loads((run / "history.json").read_text()), json.loads((run / "summary.json").read_text())
        label = f"{run.parent.name}/{run.name}"
        if i == 0:
            cols.append(("before (epoch 0, " + label + ")", history["records"][0]))
        before = history["records"][0]
        cols.append((f"{label} epoch {summary['best_epoch']}", history["records"][summary["best_epoch"]]))
        if i > 0 and abs(before["selection_score"] - cols[0][1]["selection_score"]) > 1e-9:
            cols.append((f"(before of {label}: differs from the first)", before))

    def get(rec: dict, path: str):
        cur = rec
        for key in path.split("."):
            if key not in cur:
                return None
            cur = cur[key]
        return cur

    rows = [("CV val acc", "cv_acc", 100), ("UAV val acc", "uav_acc", 100),
            ("FASDD-RS val recall", "rs_fasdd.recall", 100), ("FASDD-RS val false-alarm", "rs_fasdd.false_alarm_rate", 100),
            ("FASDD-RS val balanced acc", "rs_fasdd.balanced_accuracy", 100), ("FASDD-RS val ROC-AUC", "rs_fasdd.roc_auc", 1),
            ("FASDD-RS val `both` recall", "rs_fasdd_both_recall", 100),
            ("plume val recall", "rs_plume.recall", 100), ("plume val false-alarm", "rs_plume.false_alarm_rate", 100),
            ("plume val balanced acc", "rs_plume.balanced_accuracy", 100), ("plume val ROC-AUC", "rs_plume.roc_auc", 1),
            ("pooled RS binary balanced acc", "rs_pooled_binary_ba", 100), ("selection score", "selection_score", 100)]
    width = max(len(c[0]) for c in cols) + 2
    print(f"{'':32s}" + "".join(f"{c[0]:>{width}s}" for c in cols))
    for name, path, scale in rows:
        cells = []
        for _, rec in cols:
            v = get(rec, path)
            cells.append("-" if v is None else (f"{scale * v:.1f}" if scale == 100 else f"{v:.3f}"))
        print(f"{name:32s}" + "".join(f"{c:>{width}s}" for c in cells))
    print("\n(percent, except ROC-AUC; validation sets, not test results)")


if __name__ == "__main__":
    main()
