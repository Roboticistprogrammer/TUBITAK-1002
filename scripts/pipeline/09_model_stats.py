"""Parameter count, GFLOPs and artifact sizes for every model (issue 1.3).

Reports weights-only numbers, never the size of a training checkpoint (a `last.pt`/`best.pt` also stores the AdamW
state, about 3x the weights). Any mix of inputs can be given; each becomes one row.

    python scripts/pipeline/09_model_stats.py \
        --onnx artifacts/retrain/kd_uniform_seed42.onnx \
        --engine artifacts/engines/kd_uniform_seed42_fp16.engine artifacts/engines/kd_uniform_seed42_int8.engine \
        --checkpoint outputs/retrain/kd_uniform/seed42/best.pt outputs/teachers/FASDD_CV/best.pt

- ONNX: parameters = sum of initializer element counts (needs the `onnx` package), file size in bytes/MiB.
- Engine: file size only (TensorRT engines do not expose a parameter count).
- Checkpoint: parameters of the restored model, weights-only fp32 size, and GFLOPs of one 3x192x192 forward pass
  (torch.utils.flop_counter: 2 x multiply-accumulates of matmuls and convolutions; attention softmax and
  element-wise ops are not counted). Needs `transformers` and torch, so run it on the training machine.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import setup_path  # noqa: E402

setup_path()


def file_row(path: Path, kind: str) -> dict:
    size = path.stat().st_size
    return {"name": path.name, "kind": kind, "file_bytes": size, "file_mib": round(size / 1024**2, 2)}


def onnx_row(path: Path) -> dict:
    import onnx

    model = onnx.load(str(path), load_external_data=True)
    params = sum(int(np.prod(tensor.dims)) for tensor in model.graph.initializer)
    row = file_row(path, "onnx")
    row.update({"parameters": params, "opset": [o.version for o in model.opset_import], "nodes": len(model.graph.node),
                "fp32_weights_mib": round(params * 4 / 1024**2, 2)})
    return row


def count_gflops(model, img_size: int, device) -> float:
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    with torch.inference_mode(), FlopCounterMode(display=False) as counter:
        model(torch.zeros(1, 3, img_size, img_size, device=device))
    return counter.get_total_flops() / 1e9


def checkpoint_row(path: Path, img_size: int) -> dict:
    import torch

    from firecls.deployment.model import load_checkpoint_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, metadata = load_checkpoint_model(path, device)
    params = sum(p.numel() for p in model.parameters())
    row = file_row(path, "checkpoint")
    row.update({"parameters": params, "fp32_weights_mib": round(params * 4 / 1024**2, 2), "gflops_per_image": round(count_gflops(model, img_size, device), 2),
                "model_name": metadata["model_name"], "epoch": metadata["epoch"],
                "note": "file_mib includes optimizer state if present; fp32_weights_mib is the weights-only size"})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, nargs="*", default=[])
    parser.add_argument("--engine", type=Path, nargs="*", default=[])
    parser.add_argument("--checkpoint", type=Path, nargs="*", default=[])
    parser.add_argument("--img-size", type=int, default=192)
    parser.add_argument("--output", type=Path, default=Path("results/model_stats.json"))
    args = parser.parse_args()

    rows = [onnx_row(p) for p in args.onnx] + [file_row(p, "tensorrt_engine") for p in args.engine]
    rows += [checkpoint_row(p, args.img_size) for p in args.checkpoint]
    if not rows:
        raise SystemExit("give at least one of --onnx / --engine / --checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("| artifact | kind | parameters (M) | GFLOPs | file MiB | fp32 weights MiB |\n|---|---|---|---|---|---|")
    for r in rows:
        params = f"{r['parameters'] / 1e6:.2f}" if "parameters" in r else "-"
        print(f"| {r['name']} | {r['kind']} | {params} | {r.get('gflops_per_image', '-')} | {r['file_mib']} | {r.get('fp32_weights_mib', '-')} |")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
