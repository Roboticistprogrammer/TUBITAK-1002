from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

DEFAULT_MODEL_NAME = "microsoft/swinv2-large-patch4-window12-192-22k"
DEFAULT_IMG_SIZE = 192
DEFAULT_CLASSES = ["fire", "smoke", "both", "neither"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    tdml_json: Path
    images_root: Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_dataset_specs(root: Path | None = None) -> Dict[str, DatasetSpec]:
    repo_root = root or get_repo_root()
    return {
        "cv": DatasetSpec(
            name="FASDD_CV",
            tdml_json=repo_root
            / "datasets"
            / "FASDD_CV"
            / "FASDD_CV"
            / "annotations"
            / "TDML_CV"
            / "FASDD_CV_TDML.json",
            images_root=repo_root / "datasets" / "FASDD_CV" / "FASDD_CV" / "images",
        ),
        "rs": DatasetSpec(
            name="FASDD_RS",
            tdml_json=repo_root
            / "datasets"
            / "FASDD_RS"
            / "annotations"
            / "TDML_RS_RGB"
            / "FASDD_RS_RGB_TDML.json",
            images_root=repo_root / "datasets" / "FASDD_RS" / "images",
        ),
        "uav": DatasetSpec(
            name="FASDD_UAV",
            tdml_json=repo_root
            / "datasets"
            / "FASDD_UAV"
            / "annotations"
            / "TDML_UAV"
            / "FASDD_UAV_TDML.json",
            images_root=repo_root / "datasets" / "FASDD_UAV" / "images",
        ),
    }
