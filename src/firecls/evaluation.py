from __future__ import annotations

from typing import Sequence

import numpy as np


def classification_metrics(
    labels: Sequence[int], predictions: Sequence[int], class_names: Sequence[str]
) -> dict:
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    class_count = len(class_names)
    counts = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(counts, (labels_array, predictions_array), 1)
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=row_totals != 0,
    )

    support = counts.sum(axis=1)
    predicted_support = counts.sum(axis=0)
    true_positive = np.diag(counts)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros(class_count, dtype=np.float64),
        where=predicted_support != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(class_count, dtype=np.float64),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(class_count, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    present = support > 0
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(class_names)
    }

    return {
        "samples": int(labels_array.size),
        "accuracy": float(true_positive.sum() / max(1, labels_array.size)),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_precision_present_classes": float(precision[present].mean()),
        "macro_recall_present_classes": float(recall[present].mean()),
        "macro_f1_present_classes": float(f1[present].mean()),
        "per_class": per_class,
        "confusion_matrix": counts.tolist(),
        "confusion_matrix_normalized": normalized.tolist(),
    }


def prediction_agreement(reference_logits: np.ndarray, candidate_logits: np.ndarray) -> dict:
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError(
            f"Logit shape mismatch: reference={reference_logits.shape}, candidate={candidate_logits.shape}"
        )
    reference_predictions = reference_logits.argmax(axis=1)
    candidate_predictions = candidate_logits.argmax(axis=1)
    absolute_error = np.abs(reference_logits - candidate_logits)
    changed = int(np.count_nonzero(reference_predictions != candidate_predictions))
    return {
        "samples": int(reference_logits.shape[0]),
        "top1_agreement": float(np.mean(reference_predictions == candidate_predictions)),
        "changed_predictions": changed,
        "mean_absolute_logit_error": float(absolute_error.mean()),
        "max_absolute_logit_error": float(absolute_error.max()),
    }
