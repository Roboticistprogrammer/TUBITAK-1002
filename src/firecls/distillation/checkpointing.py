from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(checkpoint_path: Path, payload: Dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(checkpoint_path, map_location=map_location)
