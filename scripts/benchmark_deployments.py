from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import torch

from bootstrap import setup_path

ROOT = setup_path()

from firecls.config import DEFAULT_IMG_SIZE
from firecls.deployment.artifacts import git_commit, sha256_file, write_json
from firecls.deployment.model import load_checkpoint_model
from firecls.deployment.runners import OnnxRuntimeRunner, PyTorchRunner, TensorRTRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark identical student runtimes in one environment.")
    parser.add_argument("--student-pt", type=Path, required=True)
    parser.add_argument("--student-onnx", type=Path)
    parser.add_argument("--student-engine", type=Path)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("results/deployment_benchmark.json"))
    return parser.parse_args()


def benchmark_runner(runner, batch_sizes: list[int], warmup: int, runs: int) -> dict:
    results = {}
    generator = torch.Generator().manual_seed(42)
    for batch_size in batch_sizes:
        images = torch.randn(
            batch_size, 3, DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE, generator=generator
        )
        for _ in range(warmup):
            runner.infer(images)
        timings = []
        for _ in range(runs):
            started = time.perf_counter()
            runner.infer(images)
            timings.append((time.perf_counter() - started) * 1000.0)
        values = np.asarray(timings)
        mean_batch_ms = float(values.mean())
        results[str(batch_size)] = {
            "batch_size": batch_size,
            "runs": runs,
            "mean_batch_ms": mean_batch_ms,
            "median_batch_ms": float(np.median(values)),
            "p95_batch_ms": float(np.percentile(values, 95)),
            "std_batch_ms": float(values.std()),
            "mean_ms_per_image": mean_batch_ms / batch_size,
            "throughput_images_per_second": 1000.0 * batch_size / mean_batch_ms,
        }
    return results


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, metadata = load_checkpoint_model(args.student_pt, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    runners = {
        "student_pt": (PyTorchRunner(model, device), args.student_pt),
    }
    if args.student_onnx:
        runners["student_onnx"] = (
            OnnxRuntimeRunner(args.student_onnx, use_cuda=device.type == "cuda"),
            args.student_onnx,
        )
    if args.student_engine:
        runners["student_engine"] = (TensorRTRunner(args.student_engine, device), args.student_engine)

    payload = {
        "protocol": {
            "input": [3, DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE],
            "warmup_runs": args.warmup,
            "timed_runs": args.runs,
            "timing_scope": "runtime infer call including tensor transfer and output synchronization; preprocessing excluded",
            "git_commit": git_commit(ROOT),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "model": {
            "model_name": metadata["model_name"],
            "classes": metadata["classes"],
            "parameter_count": parameter_count,
            "inference_parameter_mib": parameter_bytes / (1024**2),
            "training_checkpoint_mib": args.student_pt.stat().st_size / (1024**2),
        },
        "runtimes": {},
    }
    for name, (runner, artifact) in runners.items():
        print(f"Benchmarking {name} on {device}...")
        payload["runtimes"][name] = {
            "artifact": str(artifact),
            "artifact_sha256": sha256_file(artifact),
            "artifact_mib": artifact.stat().st_size / (1024**2),
            "batches": benchmark_runner(runner, args.batch_sizes, args.warmup, args.runs),
        }
    write_json(args.output, payload)
    print(f"Saved deployment benchmark: {args.output}")


if __name__ == "__main__":
    main()
