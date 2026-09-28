"""Plug-in registry so shared scripts can load any baseline family's PyTorch checkpoint.

A family lives in ``firecls/baselines/families/<name>.py`` and decorates one loader:

    @register_family("cnn")
    def load(checkpoint: Path, device) -> LoadedBaseline: ...

The loader must return a module whose forward maps a preprocessed batch to ``[batch, 4]``
scores ordered like ``protocol.CLASSES``. Families are discovered automatically, so each
baseline branch only adds its own module and nothing else needs editing.
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

_LOADERS: dict[str, Callable[..., "LoadedBaseline"]] = {}


@dataclass
class LoadedBaseline:
    model: torch.nn.Module  # forward(images) -> [batch, 4] scores
    preprocessing: dict  # see firecls.baselines.preprocessing
    metadata: dict = field(default_factory=dict)  # must include "classes" and "model_name"


def register_family(name: str):
    def decorator(loader):
        if name in _LOADERS:
            raise ValueError(f"Baseline family '{name}' registered twice.")
        _LOADERS[name] = loader
        return loader

    return decorator


def _discover() -> None:
    from firecls.baselines import families

    for module in pkgutil.iter_modules(families.__path__):
        importlib.import_module(f"{families.__name__}.{module.name}")


def available_families() -> list[str]:
    _discover()
    return sorted(_LOADERS)


def load_baseline(family: str, checkpoint: Path, device: torch.device | str = "cpu") -> LoadedBaseline:
    _discover()
    if family not in _LOADERS:
        raise KeyError(f"Unknown baseline family '{family}'. Available: {sorted(_LOADERS)}")
    loaded = _LOADERS[family](Path(checkpoint), device)
    loaded.model.to(device).eval()
    return loaded
