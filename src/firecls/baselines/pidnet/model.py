"""PIDNet-S construction and the image-level scoring wrapper exported to ONNX/TensorRT.

The segmentation network predicts two independent sigmoid channels (fire, smoke). The thesis
contract needs one ``[batch, 4]`` score vector per image, so :class:`PidnetPresenceScores`
reduces each channel's probability map to a *presence* probability and feeds both into the
shared :class:`~firecls.baselines.scores.PresenceToScores` head. Everything, including ImageNet
normalisation, is inside the module so the exported graph consumes the letterboxed ``[0, 1]``
tensor produced by ``firecls.baselines.preprocessing.Letterbox``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F

from firecls.baselines.preprocessing import IMAGENET_MEAN, IMAGENET_STD
from firecls.baselines.scores import PresenceToScores
from firecls.config import get_repo_root

SEG_CHANNELS: list[str] = ["fire", "smoke"]
MODEL_NAME = "pidnet_s"
OUTPUT_STRIDE = 8  # PIDNet heads operate at 1/8 of the input resolution
# Upstream ``get_seg_model`` arguments for the "s" variant.
PIDNET_S_KWARGS = dict(m=2, n=3, planes=32, ppm_planes=96, head_planes=128)
# TensorRT 8.x limits TopK to K <= 3840; TensorRT 10 raised it, but Jetson Nano/Xavier stay on 8.x.
TENSORRT_TOPK_LIMIT = 3840
AUX_HEAD_PREFIXES = ("seghead_p.", "seghead_d.")

_VENDORED_NAME = "_firecls_vendored_pidnet"


def vendored_pidnet() -> ModuleType:
    """Import ``third_party/pidnet`` under a private name (no sys.path mutation, no name clash)."""
    if _VENDORED_NAME in sys.modules:
        return sys.modules[_VENDORED_NAME]
    package_dir = get_repo_root() / "third_party" / "pidnet"
    spec = importlib.util.spec_from_file_location(
        _VENDORED_NAME, package_dir / "__init__.py", submodule_search_locations=[str(package_dir)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Vendored PIDNet not found in {package_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_VENDORED_NAME] = module
    spec.loader.exec_module(module)
    return module


def build_pidnet_s(num_classes: int = len(SEG_CHANNELS), augment: bool = True) -> torch.nn.Module:
    """PIDNet-S. ``augment=True`` adds the auxiliary P and D heads needed for training."""
    return vendored_pidnet().PIDNet(num_classes=num_classes, augment=augment, **PIDNET_S_KWARGS)


def load_imagenet_pretrained(model: torch.nn.Module, path: Path) -> dict:
    """Copy name- and shape-matching tensors from ``PIDNet_S_ImageNet.pth.tar`` (upstream rule)."""
    from firecls.deployment.artifacts import sha256_file

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # older pickles may hold non-tensor objects; the file is user-supplied
        payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    own = model.state_dict()
    matched = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    if not matched:
        raise ValueError(f"No PIDNet-S tensors matched in {path}; is this PIDNet_S_ImageNet.pth.tar?")
    own.update(matched)
    model.load_state_dict(own)
    return {"path": str(path), "sha256": sha256_file(Path(path)), "tensors_loaded": len(matched), "tensors_total": len(own)}


def inference_state_dict(state: dict) -> dict:
    """Drop the auxiliary training heads so the state loads into an ``augment=False`` PIDNet."""
    return {k: v for k, v in state.items() if not k.startswith(AUX_HEAD_PREFIXES)}


def output_size(img_size: int) -> int:
    if img_size % OUTPUT_STRIDE:
        raise ValueError(f"PIDNet input size must be a multiple of {OUTPUT_STRIDE}, got {img_size}")
    return img_size // OUTPUT_STRIDE


def topk_count(img_size: int, topk_fraction: float) -> int:
    """K = max(1, round(fraction * H_out * W_out)) on the 1/8-resolution logit map."""
    if not 0.0 < topk_fraction <= 1.0:
        raise ValueError("topk_fraction must lie in (0, 1].")
    side = output_size(img_size)
    k = max(1, round(topk_fraction * side * side))
    if k > TENSORRT_TOPK_LIMIT:
        raise ValueError(
            f"K={k} exceeds the TensorRT 8.x TopK limit ({TENSORRT_TOPK_LIMIT}); lower --topk-fraction or --img-size."
        )
    return k


def presence_from_logits(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean of the ``k`` highest pixel probabilities -> ``(p_fire, p_smoke)``.

    A mean over the top-K pixels instead of the single maximum makes the image decision robust
    to isolated hot pixels, while K stays small enough that a small fire still counts.
    """
    probabilities = torch.sigmoid(logits.float()).flatten(2)  # [B, 2, H*W]
    presence = probabilities.topk(k, dim=2).values.mean(dim=2)  # [B, 2]
    return presence[:, 0], presence[:, 1]


class PidnetPresenceScores(torch.nn.Module):
    """``[B,3,S,S]`` letterboxed image in ``[0,1]`` -> ``[B,4]`` scores (arg-max = label)."""

    def __init__(
        self,
        net: torch.nn.Module,
        img_size: int,
        topk_fraction: float,
        t_fire: float = 0.5,
        t_smoke: float = 0.5,
    ) -> None:
        super().__init__()
        self.net = net
        self.img_size = int(img_size)
        self.topk_fraction = float(topk_fraction)
        self.k = topk_count(self.img_size, self.topk_fraction)
        self.head = PresenceToScores(t_fire, t_smoke)
        # Non-persistent: constants in the ONNX graph, absent from the saved state dict.
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def normalise(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean) / self.std

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        """Main-branch logits at 1/8 resolution, ``[B, 2, S/8, S/8]``."""
        outputs = self.net(self.normalise(images))
        if isinstance(outputs, (list, tuple)):  # augment=True network: [P, main, D]
            outputs = outputs[1]
        return outputs

    def presence(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return presence_from_logits(self.logits(images), self.k)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        p_fire, p_smoke = self.presence(images)
        return self.head(p_fire, p_smoke)

    def export_module(self) -> torch.nn.Module:
        """Hook used by ``scripts/export_baseline.py``; the wrapper is already export-ready."""
        return self.eval()


def upsample_like(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinear upsampling to label resolution, as upstream ``FullModel`` does before the loss."""
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
