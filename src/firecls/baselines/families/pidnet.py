"""PIDNet-S trained on SAM box-prompted pseudo-masks (re-implementation of Pesonen et al., WACV 2025).

The segmenter's two sigmoid channels are reduced to image-level presence by a top-K mean and
mapped to the 4-way scores with the validation-calibrated thresholds from ``thresholds.json``
beside the checkpoint (see ``firecls.baselines.pidnet.model``).
"""
from __future__ import annotations

from pathlib import Path

from firecls.baselines.pidnet.checkpoint import (
    load_checkpoint,
    metadata_from_checkpoint,
    read_thresholds,
    scorer_from_checkpoint,
)
from firecls.baselines.preprocessing import letterbox_spec
from firecls.baselines.registry import LoadedBaseline, register_family


@register_family("pidnet")
def load(checkpoint: Path, device) -> LoadedBaseline:
    payload = load_checkpoint(checkpoint)
    thresholds, calibrated = read_thresholds(checkpoint)
    model = scorer_from_checkpoint(payload, thresholds["t_fire"], thresholds["t_smoke"])
    preprocessing = letterbox_spec(int(payload["img_size"]), int(payload.get("pad_value", 114)))
    return LoadedBaseline(model, preprocessing, metadata_from_checkpoint(payload, thresholds, calibrated))
