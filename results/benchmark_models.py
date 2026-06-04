"""
Benchmark model inference performance on host machine.
Measures: latency, FPS, throughput, model size, peak GPU memory.
Supports: .pt, .onnx, .engine
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
from ultralytics import YOLO


def benchmark_model(model_path: Path, model_type: str, batch_sizes: list = [1], num_runs: int = 100) -> Dict:
    """Benchmark any model format via Ultralytics inference."""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {model_path.name} ({model_type})")
    print(f"{'='*70}")

    model = YOLO(str(model_path))
    dummy_image = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)

    results = {}
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch_size={batch_size}")
        source = [dummy_image.copy() for _ in range(batch_size)]

        for _ in range(10):
            _ = model.predict(source=source, imgsz=640, conf=0.25, iou=0.6, verbose=False)

        timings_ms = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model.predict(source=source, imgsz=640, conf=0.25, iou=0.6, verbose=False)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            timings_ms.append(elapsed_ms)

        avg_ms = float(np.mean(timings_ms))
        std_ms = float(np.std(timings_ms))
        per_image_ms = avg_ms / max(batch_size, 1)
        fps = float(1000.0 / per_image_ms) if per_image_ms > 0 else 0.0

        results[f"batch_{batch_size}"] = {
            "latency_ms": per_image_ms,
            "std_ms": std_ms,
            "fps": fps,
            "throughput": fps,
            "batch_latency_ms": avg_ms,
        }

        print(f"  Latency/image: {per_image_ms:.2f} ms")
        print(f"  FPS: {fps:.1f}")

    print(f"\n{'='*70}")
    model_size_mb = model_path.stat().st_size / (1024 ** 2)
    results["model_size_mb"] = float(model_size_mb)
    print(f"Model size: {results['model_size_mb']:.2f} MB")
    print(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference on host machine.")
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to model file (.pt, .onnx, or .engine)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["pt", "onnx", "engine"],
        help="Model format type"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Batch sizes to benchmark (default: 1)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of benchmark runs (default: 100)"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/benchmark_results.json"),
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    if not args.model_path:
        print("Error: --model-path is required")
        parser.print_help()
        return
    
    if not args.model_path.exists():
        print(f"Error: Model not found at {args.model_path}")
        return
    
    # Benchmark
    results = benchmark_model(args.model_path, args.model_type, args.batch_sizes, num_runs=args.num_runs)
    
    # Extract batch_1 results for main comparison
    batch_1_results = results.get("batch_1", {})
    output_data = {
        "model_name": args.model_path.stem,
        "model_type": args.model_type,
        "deployment": {
            "latency_ms": batch_1_results.get("latency_ms", 0.0),
            "fps": batch_1_results.get("fps", 0.0),
            "throughput": batch_1_results.get("throughput", 0.0),
            "size_mb": results.get("model_size_mb", 0.0),
        },
        "all_batch_results": results,
    }
    
    if "peak_gpu_memory_mb" in results:
        output_data["deployment"]["peak_gpu_mem_mb"] = results["peak_gpu_memory_mb"]
    
    # Save result
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
