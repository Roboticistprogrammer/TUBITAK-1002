from __future__ import annotations

import argparse
import platform
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bootstrap import setup_path

ROOT = setup_path()

from firecls.config import DEFAULT_IMG_SIZE
from firecls.deployment.artifacts import git_commit, sha256_file, write_json
from firecls.deployment.model import load_checkpoint_model


class LogitsOnly(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values)
        return getattr(outputs, "logits", outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the distilled Microsoft SwinV2 student to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/student.onnx"))
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fixed-batch", action="store_true")
    parser.add_argument("--verification-atol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model, processor, metadata = load_checkpoint_model(args.checkpoint, "cpu")
    if len(metadata["classes"]) != 4:
        raise ValueError(f"Expected four student classes, found {metadata['classes']}")
    if args.img_size != DEFAULT_IMG_SIZE:
        raise ValueError(
            f"The trained student expects {DEFAULT_IMG_SIZE}x{DEFAULT_IMG_SIZE}; got {args.img_size}."
        )

    wrapper = LogitsOnly(model).eval()
    torch.manual_seed(42)
    dummy = torch.randn(1, 3, args.img_size, args.img_size, dtype=torch.float32)
    dynamic_axes = None
    if not args.fixed_batch:
        dynamic_axes = {"pixel_values": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    import onnxruntime as ort

    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    with torch.inference_mode():
        reference = wrapper(dummy).cpu().numpy()
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    candidate = session.run(["logits"], {"pixel_values": dummy.numpy()})[0]
    max_error = float(np.max(np.abs(reference - candidate)))
    top1_matches = bool(reference.argmax(axis=1)[0] == candidate.argmax(axis=1)[0])
    if not top1_matches or max_error > args.verification_atol:
        raise RuntimeError(
            f"ONNX verification failed: top1_match={top1_matches}, max_abs_error={max_error:.8f}"
        )

    manifest = {
        "artifact": str(args.output),
        "artifact_sha256": sha256_file(args.output),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "source_git_commit": git_commit(ROOT),
        "model_name": metadata["model_name"],
        "classes": metadata["classes"],
        "checkpoint_epoch": metadata["epoch"],
        "checkpoint_avg_val": metadata["avg_val"],
        "input": {"name": "pixel_values", "shape": ["batch", 3, args.img_size, args.img_size]},
        "output": {"name": "logits", "shape": ["batch", len(metadata["classes"])]},
        "preprocessing": {
            "resize": args.img_size + 32,
            "center_crop": args.img_size,
            "mean": processor.image_mean,
            "std": processor.image_std,
        },
        "opset": args.opset,
        "verification": {"top1_match": top1_matches, "max_absolute_logit_error": max_error},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    write_json(manifest_path, manifest)
    print(f"Exported verified four-class student: {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
