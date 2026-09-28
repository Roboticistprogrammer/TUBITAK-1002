"""Checkpoint and ``thresholds.json`` I/O shared by the scripts and the ``pidnet`` family loader.

Checkpoints hold only tensors and plain Python containers, so they load with
``torch.load(weights_only=True)``. ``thresholds.json`` records the SHA-256 of the checkpoint it
was calibrated for; a mismatch (e.g. re-training overwrote ``best.pt``) is an error rather than
a silent use of stale thresholds.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch

from firecls.baselines.pidnet.model import (
    MODEL_NAME,
    SEG_CHANNELS,
    PidnetPresenceScores,
    build_pidnet_s,
    inference_state_dict,
)
from firecls.baselines.protocol import CLASSES
from firecls.deployment.artifacts import sha256_file, write_json

THRESHOLDS_FILE = "thresholds.json"
DEFAULT_THRESHOLDS = {"t_fire": 0.5, "t_smoke": 0.5}


def load_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("classes") != CLASSES:
        raise ValueError(f"{path}: class order {payload.get('classes')} differs from protocol {CLASSES}")
    if payload.get("seg_channels", SEG_CHANNELS) != SEG_CHANNELS:
        raise ValueError(f"{path}: segmentation channels {payload.get('seg_channels')} != {SEG_CHANNELS}")
    return payload


def scorer_from_checkpoint(payload: dict, t_fire: float = 0.5, t_smoke: float = 0.5) -> PidnetPresenceScores:
    """Inference-only PIDNet-S (auxiliary heads dropped) wrapped into the ``[B, 4]`` contract."""
    net = build_pidnet_s(len(SEG_CHANNELS), augment=False)
    net.load_state_dict(inference_state_dict(payload["model"]), strict=True)
    return PidnetPresenceScores(net, payload["img_size"], payload["topk_fraction"], t_fire, t_smoke).eval()


def thresholds_path(checkpoint: Path) -> Path:
    return Path(checkpoint).parent / THRESHOLDS_FILE


def write_thresholds(checkpoint: Path, payload: dict) -> Path:
    path = thresholds_path(checkpoint)
    write_json(path, {**payload, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint)})
    return path


def read_thresholds(checkpoint: Path) -> tuple[dict, bool]:
    """``({"t_fire", "t_smoke"}, calibrated)``; falls back to 0.5/0.5 with a warning."""
    path = thresholds_path(checkpoint)
    if not path.exists():
        warnings.warn(
            f"{path} not found: using uncalibrated thresholds 0.5/0.5. Run scripts/pidnet/calibrate_thresholds.py "
            "before reporting results.",
            stacklevel=2,
        )
        return dict(DEFAULT_THRESHOLDS), False
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("checkpoint_sha256")
    if expected and expected != sha256_file(checkpoint):
        raise ValueError(f"{path} was calibrated for a different checkpoint than {checkpoint}; re-run calibration.")
    return {"t_fire": float(payload["t_fire"]), "t_smoke": float(payload["t_smoke"])}, True


def metadata_from_checkpoint(payload: dict, thresholds: dict, calibrated: bool) -> dict:
    return {
        "classes": list(CLASSES),
        "model_name": payload.get("model_name", MODEL_NAME),
        "source": "reimplemented",
        "thresholds": dict(thresholds),
        "thresholds_calibrated": calibrated,
        "seg_channels": list(SEG_CHANNELS),
        "img_size": int(payload["img_size"]),
        "topk_fraction": float(payload["topk_fraction"]),
        "epoch": payload.get("epoch"),
        "train_seed": payload.get("seed"),
        "select_by": payload.get("select_by"),
        "init": payload.get("init"),
    }
