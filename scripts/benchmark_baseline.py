"""Latency, throughput and (on Jetson) power for one baseline, per runtime and batch size.

Timing covers the runtime ``infer`` call including host<->device transfer; preprocessing and
image decoding are excluded, identical to ``scripts/benchmark_deployments.py``. On Jetson,
``tegrastats`` is sampled during the timed loop and energy per image is derived from it.

Run on the Jetson with ``sudo jetson_clocks`` and a fixed ``nvpmodel`` mode, and record both
in the thesis. For a JetPack whose Python cannot run this script (e.g. JetPack 4 on Nano),
use ``scripts/jetson_trtexec_benchmark.sh`` instead.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from bootstrap import setup_path

ROOT = setup_path()

from firecls.baselines.power import TegrastatsMonitor
from firecls.baselines.preprocessing import input_size
from firecls.baselines.registry import load_baseline
from firecls.deployment.artifacts import git_commit, sha256_file, write_json
from firecls.deployment.runners import OnnxRuntimeRunner, PyTorchRunner, TensorRTRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", type=Path, help="PyTorch checkpoint (optional on Jetson)")
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def time_runner(runner, size: int, batch_sizes: list[int], warmup: int, runs: int, log_dir: Path) -> dict:
    results = {}
    generator = torch.Generator().manual_seed(42)
    for batch in batch_sizes:
        images = torch.rand(batch, 3, size, size, generator=generator)
        for _ in range(warmup):
            runner.infer(images)
        timings = []
        with TegrastatsMonitor(log_path=log_dir / f"tegrastats_b{batch}.log") as monitor:
            for _ in range(runs):
                start = time.perf_counter()
                runner.infer(images)
                timings.append((time.perf_counter() - start) * 1000.0)
        values = np.asarray(timings)
        mean_ms = float(values.mean())
        entry = {
            "batch_size": batch,
            "runs": runs,
            "mean_batch_ms": mean_ms,
            "p50_batch_ms": float(np.percentile(values, 50)),
            "p95_batch_ms": float(np.percentile(values, 95)),
            "p99_batch_ms": float(np.percentile(values, 99)),
            "std_batch_ms": float(values.std()),
            "mean_ms_per_image": mean_ms / batch,
            "throughput_images_per_second": 1000.0 * batch / mean_ms,
        }
        power = monitor.summary()
        if power:
            entry["power"] = power
            entry["energy_mj_per_image"] = power["mean_mw"] * (mean_ms / batch) / 1000.0
        results[str(batch)] = entry
        print(f"  batch={batch}: {mean_ms:.2f} ms/batch, {entry['throughput_images_per_second']:.1f} img/s")
    return results


def manifest_size(artifact: Path) -> int:
    manifest = json.loads(artifact.with_suffix(artifact.suffix + ".manifest.json").read_text(encoding="utf-8"))
    return int(manifest["input"]["shape"][-1])


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runners: dict[str, tuple] = {}
    model_info: dict = {}
    size = None
    if args.checkpoint:
        loaded = load_baseline(args.family, args.checkpoint, device)
        size = input_size(loaded.preprocessing)
        model_info = {
            "model_name": loaded.metadata.get("model_name"),
            "parameter_count": int(sum(p.numel() for p in loaded.model.parameters())),
        }
        runners["pt"] = (PyTorchRunner(loaded.model, device), args.checkpoint)
    if args.onnx:
        size = size or manifest_size(args.onnx)
        runners["onnx"] = (OnnxRuntimeRunner(args.onnx, use_cuda=device.type == "cuda"), args.onnx)
    if args.engine:
        size = size or manifest_size(args.engine)
        runners["engine"] = (TensorRTRunner(args.engine, device), args.engine)
    if not runners:
        raise SystemExit("Provide at least one of --checkpoint, --onnx, --engine.")

    log_dir = args.output.parent / f"{args.name}_tegrastats"
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": {"name": args.name, "family": args.family, **model_info},
        "protocol": {
            "input": [3, size, size],
            "warmup_runs": args.warmup,
            "timed_runs": args.runs,
            "timing_scope": "runtime infer call incl. H2D/D2H transfer; preprocessing excluded",
            "git_commit": git_commit(ROOT),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "runtimes": {},
    }
    for runtime, (runner, artifact) in runners.items():
        print(f"Benchmarking {args.name}/{runtime} on {device}")
        payload["runtimes"][runtime] = {
            "artifact": str(artifact),
            "artifact_sha256": sha256_file(artifact),
            "artifact_mib": artifact.stat().st_size / 1024**2,
            "batches": time_runner(runner, size, args.batch_sizes, args.warmup, args.runs, log_dir),
        }
    write_json(args.output, payload)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
