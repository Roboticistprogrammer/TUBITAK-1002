"""ONNX export with a verification step and a sidecar manifest.

The manifest format is the one ``scripts/build_engine.py`` already consumes for the student,
so every baseline goes through the identical ONNX -> TensorRT FP16 path.
"""
from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import torch

from firecls.baselines.preprocessing import input_size
from firecls.baselines.protocol import CLASSES
from firecls.config import get_repo_root
from firecls.deployment.artifacts import git_commit, sha256_file, write_json

INPUT_NAME = "pixel_values"
OUTPUT_NAME = "logits"  # kept for compatibility with the student runners; values are scores


def export_scores_model(
    model: torch.nn.Module,
    output: Path,
    preprocessing: dict,
    source_checkpoint: Path,
    family: str,
    model_name: str,
    source: str,
    opset: int = 17,
    dynamic_batch: bool = True,
    atol: float = 1e-3,
    extra: dict | None = None,
) -> Path:
    """Export ``model`` (``[B,3,H,W] -> [B,4]``), verify against ONNX Runtime, write manifest."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    size = input_size(preprocessing)
    model = model.eval().cpu()
    torch.manual_seed(0)
    dummy = torch.rand(2, 3, size, size) if preprocessing["mode"] == "letterbox" else torch.randn(2, 3, size, size)
    dynamic_axes = {INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}} if dynamic_batch else None

    export_kwargs = dict(
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(model, dummy, output, dynamo=False, **export_kwargs)
    except TypeError:  # torch < 2.5 has no ``dynamo`` argument
        torch.onnx.export(model, dummy, output, **export_kwargs)

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(output))
    with torch.inference_mode():
        reference = model(dummy).cpu().numpy()
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    candidate = session.run([OUTPUT_NAME], {INPUT_NAME: dummy.numpy()})[0]
    if reference.shape != (dummy.shape[0], len(CLASSES)):
        raise RuntimeError(f"Model must return [batch, {len(CLASSES)}] scores, got {reference.shape}")
    max_error = float(np.max(np.abs(reference - candidate)))
    top1 = bool(np.array_equal(reference.argmax(1), candidate.argmax(1)))
    if not top1 or max_error > atol:
        raise RuntimeError(f"ONNX verification failed: top1_match={top1}, max_abs_error={max_error:.6f}")

    manifest = {
        "artifact": str(output),
        "artifact_sha256": sha256_file(output),
        "family": family,
        "model_name": model_name,
        "source": source,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "source_git_commit": git_commit(get_repo_root()),
        "classes": CLASSES,
        "input": {"name": INPUT_NAME, "shape": ["batch", 3, size, size]},
        "output": {"name": OUTPUT_NAME, "shape": ["batch", len(CLASSES)], "semantics": "class scores, arg-max = label"},
        "preprocessing": preprocessing,
        "opset": opset,
        "verification": {"top1_match": top1, "max_absolute_error": max_error},
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
    }
    if extra:
        manifest.update(extra)
    write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    return output
