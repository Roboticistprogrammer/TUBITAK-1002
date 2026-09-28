"""Ultralytics YOLO detectors (YOLOv5nu / YOLOv8n / YOLO11n) reduced to image-level scores.

Adaptation of Vazquez et al. (2025): the detectors are trained on FASDD fire/smoke boxes, and
the image-level label is derived from their dense predictions without NMS:

    p_fire  = max over all anchors of the fire class probability
    p_smoke = max over all anchors of the smoke class probability

NMS only removes boxes, it never raises the highest score, so the maximum over the raw head
output equals the confidence of the best surviving detection. Skipping NMS keeps the exported
graph free of data-dependent control flow, so the ONNX -> TensorRT path is the same as for
every other baseline. The two presence probabilities are mapped to the 4-way score space by
:class:`firecls.baselines.scores.PresenceToScores` with thresholds calibrated on the
validation split (``scripts/yolo/calibrate_thresholds.py``).

Checked against ultralytics 8.4: in eval mode ``DetectionModel.forward`` returns a tuple
``(decoded [B, 4 + nc, N], raw dict)``, while with ``Detect.export = True`` it returns the
decoded tensor alone. Class scores in the decoded tensor are already sigmoid probabilities for
the anchor-free v5u / v8 / 11 heads. We set the export flags exactly as Ultralytics' own
exporter does (fused Conv+BN, ``export=True``, ``format='onnx'``) so the PyTorch module and the
ONNX graph are the same computation, and still accept a tuple for robustness across versions.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch

from firecls.baselines.boxes import canonical_class
from firecls.baselines.preprocessing import letterbox_spec
from firecls.baselines.protocol import CLASSES, SOURCE_REIMPLEMENTED
from firecls.baselines.registry import LoadedBaseline, register_family
from firecls.baselines.scores import PresenceToScores
from firecls.deployment.artifacts import sha256_file

IMG_SIZE = 640  # native Ultralytics input; see docs/BASELINES.md section 4 (input geometry)
DETECTOR_CLASSES = ("fire", "smoke")  # class ids fixed by scripts/yolo/prepare_fasdd_yolo.py
THRESHOLDS_FILE = "thresholds.json"
DEFAULT_THRESHOLD = 0.5

# CLI architecture key -> Ultralytics weight stem (COCO init) and model yaml (scratch init).
# Ultralytics only distributes the anchor-free "u" variant of YOLOv5, see docs/baselines/YOLO_VAZQUEZ.md.
ARCHITECTURES: dict[str, dict[str, str]] = {
    "yolov5n": {"weights": "yolov5nu.pt", "yaml": "yolov5nu.yaml"},
    "yolov8n": {"weights": "yolov8n.pt", "yaml": "yolov8n.yaml"},
    "yolo11n": {"weights": "yolo11n.pt", "yaml": "yolo11n.yaml"},
}
_STEM_TO_ARCH = {Path(v[key]).stem: arch for arch, v in ARCHITECTURES.items() for key in ("weights", "yaml")}


def _detect_heads(model: torch.nn.Module) -> list[torch.nn.Module]:
    from ultralytics.nn.modules.head import Detect

    return [m for m in model.modules() if isinstance(m, Detect)]


def prepare_detector(detector: torch.nn.Module) -> torch.nn.Module:
    """FP32, eval, Conv+BN fused and the Detect head in export mode, mirroring Ultralytics' exporter."""
    from ultralytics.nn.modules.block import C2f

    detector = detector.float().eval()
    for p in detector.parameters():
        p.requires_grad_(False)
    if hasattr(detector, "fuse"):
        detector = detector.fuse(verbose=False)
    heads = _detect_heads(detector)
    if len(heads) != 1:
        raise ValueError(f"Expected exactly one Detect head, found {len(heads)}")
    for head in heads:
        if head.end2end:  # YOLOv10/YOLO26 NMS-free heads return [B, max_det, 6]; not part of this baseline
            raise NotImplementedError("End-to-end (NMS-free) detection heads are not supported.")
        head.export = True
        head.format = "onnx"
        head.dynamic = False  # the spatial size is fixed (640); the batch axis stays dynamic in the trace
        head.shape = None  # drop anchors cached for a previous input size
    for module in detector.modules():
        if isinstance(module, C2f):
            module.forward = module.forward_split  # same maths as forward(); cleaner ONNX (as Ultralytics' exporter)
    return detector


class YoloPresenceScores(torch.nn.Module):
    """``[B, 3, S, S]`` letterboxed RGB in ``[0, 1]`` -> ``[B, 4]`` scores whose arg-max is the label."""

    def __init__(
        self,
        detector: torch.nn.Module,
        t_fire: float = DEFAULT_THRESHOLD,
        t_smoke: float = DEFAULT_THRESHOLD,
        fire_index: int = 0,
        smoke_index: int = 1,
    ) -> None:
        super().__init__()
        self.detector = prepare_detector(detector)
        self.nc = int(_detect_heads(self.detector)[0].nc)
        self.fire_index = int(fire_index)
        self.smoke_index = int(smoke_index)
        self.head = PresenceToScores(t_fire, t_smoke)

    def presence(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-image ``(p_fire, p_smoke)``: the highest class probability over all anchors."""
        predictions = self.detector(images)
        if isinstance(predictions, (tuple, list)):  # eval mode without export flag: (decoded, raw)
            predictions = predictions[0]
        if not isinstance(predictions, torch.Tensor) or predictions.ndim != 3:
            raise RuntimeError(
                "Detector must return decoded predictions [B, 4 + nc, N]; was the wrapper put in train mode?"
            )
        if predictions.shape[1] != 4 + self.nc:
            raise RuntimeError(f"Expected {4 + self.nc} prediction channels, got {tuple(predictions.shape)}")
        p_fire = predictions[:, 4 + self.fire_index, :].amax(dim=1)
        p_smoke = predictions[:, 4 + self.smoke_index, :].amax(dim=1)
        return p_fire, p_smoke

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(*self.presence(images))

    def export_module(self) -> torch.nn.Module:
        """The module already is export-ready (see :func:`prepare_detector`)."""
        return self.eval()


def class_indices(names) -> tuple[int, int]:
    """Locate the fire and smoke ids in an Ultralytics ``names`` mapping, failing loudly otherwise."""
    items = list(names.items() if isinstance(names, dict) else enumerate(names))
    lookup = {canonical_class(str(name)): int(index) for index, name in items}
    if len(items) != len(DETECTOR_CLASSES) or any(c not in lookup for c in DETECTOR_CLASSES):
        raise ValueError(f"Detector classes {names!r} are not exactly {list(DETECTOR_CLASSES)}")
    return lookup["fire"], lookup["smoke"]


def read_thresholds(checkpoint: Path) -> tuple[dict, bool]:
    """Thresholds written by ``calibrate_thresholds.py`` for *this* checkpoint, or 0.5/0.5 with a warning."""
    path = Path(checkpoint).parent / THRESHOLDS_FILE
    if not path.exists():
        warnings.warn(
            f"{path} not found: using uncalibrated thresholds {DEFAULT_THRESHOLD}/{DEFAULT_THRESHOLD}. "
            "Run scripts/yolo/calibrate_thresholds.py before reporting results.",
            stacklevel=2,
        )
        return {"t_fire": DEFAULT_THRESHOLD, "t_smoke": DEFAULT_THRESHOLD}, False
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("checkpoint_sha256")
    if expected and expected != sha256_file(checkpoint):
        raise ValueError(f"{path} was calibrated for a different checkpoint than {checkpoint}; re-run calibration.")
    return {"t_fire": float(payload["t_fire"]), "t_smoke": float(payload["t_smoke"])}, True


def describe_run(train_args: dict) -> dict:
    """Architecture key and initialisation recovered from Ultralytics' saved ``train_args``."""
    model_arg = str(train_args.get("model") or "")
    stem = Path(model_arg).stem
    arch = _STEM_TO_ARCH.get(stem, stem or "yolo")
    init = "coco" if model_arg.endswith(".pt") else "scratch"
    return {"arch": arch, "init": init, "model_name": f"{arch}_{init}"}


def load_presence_model(checkpoint: Path, t_fire: float = DEFAULT_THRESHOLD, t_smoke: float = DEFAULT_THRESHOLD):
    """Load an Ultralytics ``best.pt`` into :class:`YoloPresenceScores`; returns ``(model, ckpt)``."""
    from ultralytics.nn.tasks import load_checkpoint

    detector, ckpt = load_checkpoint(str(checkpoint), device="cpu", fuse=False)
    fire_index, smoke_index = class_indices(detector.names)
    return YoloPresenceScores(detector, t_fire, t_smoke, fire_index, smoke_index), ckpt


@register_family("yolo")
def load(checkpoint: Path, device) -> LoadedBaseline:
    import ultralytics

    thresholds, calibrated = read_thresholds(checkpoint)
    model, ckpt = load_presence_model(checkpoint, thresholds["t_fire"], thresholds["t_smoke"])
    train_args = ckpt.get("train_args") or {}
    run = describe_run(train_args)
    results = ckpt.get("train_results") or {}
    metadata = {
        "classes": CLASSES,
        "model_name": run["model_name"],
        "arch": run["arch"],
        "init": run["init"],
        "source": SOURCE_REIMPLEMENTED,
        "thresholds": thresholds,
        "thresholds_calibrated": calibrated,
        "detector_classes": list(DETECTOR_CLASSES),
        "decision_rule": "p_c = max over anchors of class-c probability (pre-NMS); thresholds calibrated on val",
        "ultralytics_version": ultralytics.__version__,
        "checkpoint_ultralytics_version": ckpt.get("version"),
    }
    if train_args.get("seed") is not None:
        metadata["train_seed"] = int(train_args["seed"])
    if results.get("epoch"):  # best.pt is written right after its epoch's results row
        metadata["epoch"] = int(results["epoch"][-1])
    return LoadedBaseline(model, letterbox_spec(IMG_SIZE), metadata)
