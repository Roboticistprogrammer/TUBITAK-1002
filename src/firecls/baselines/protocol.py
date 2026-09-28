"""Single source of truth for the baseline comparison protocol.

Changing a value here changes the experiment for *every* baseline, which is the point: the
thesis states that all methods share splits, class order, training budget and selection rule.
"""
from __future__ import annotations

from firecls.config import DEFAULT_CLASSES, DEFAULT_IMG_SIZE

# Image-level label space and order. Must match the distilled student exactly.
CLASSES: list[str] = list(DEFAULT_CLASSES)  # ["fire", "smoke", "both", "neither"]
DOMAINS: list[str] = ["cv", "rs", "uav"]

# Classification backbones use the student's input geometry so latency is comparable.
CLASSIFIER_IMG_SIZE: int = DEFAULT_IMG_SIZE  # 192
CLASSIFIER_RESIZE: int = DEFAULT_IMG_SIZE + 32  # 224, then centre-crop to 192

# Training budget shared by every trainable baseline (matches train_student.py defaults).
EPOCHS: int = 20
BATCH_SIZE: int = 16
SEEDS: tuple[int, ...] = (42, 43, 44)

# Model selection: checkpoint with the best equal-weight mean validation accuracy over domains.
SELECTION_SPLIT: str = "val"
EVAL_SPLIT: str = "test"
DOMAIN_AGGREGATION: str = "equal-weight macro average over cv, rs, uav"

# Provenance markers used in the results table.
SOURCE_OFFICIAL = "official"  # authors' code or an unmodified library implementation
SOURCE_REIMPLEMENTED = "reimplemented"  # rebuilt from the paper (dagger in the thesis table)
SOURCE_REPORTED = "reported"  # numbers copied from the paper (double dagger)
SOURCES = (SOURCE_OFFICIAL, SOURCE_REIMPLEMENTED, SOURCE_REPORTED)


def protocol_dict() -> dict:
    """Serialisable description embedded in every result file."""
    return {
        "classes": CLASSES,
        "domains": DOMAINS,
        "classifier_img_size": CLASSIFIER_IMG_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seeds": list(SEEDS),
        "selection": f"best mean {SELECTION_SPLIT} accuracy over domains",
        "eval_split": EVAL_SPLIT,
        "domain_aggregation": DOMAIN_AGGREGATION,
    }
