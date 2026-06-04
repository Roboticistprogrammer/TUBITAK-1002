import argparse
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ONNX/TensorRT models on Jetson")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to .onnx or .engine model")
    parser.add_argument("--model-type", choices=["onnx", "engine"], required=True, help="Backend type")
    parser.add_argument("--sample-dir", type=Path, default=Path("sample"), help="Sample images root directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"), help="Benchmark output root")
    parser.add_argument("--input-size", type=int, default=224, help="Input square size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmark")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Timed benchmark iterations")
    parser.add_argument("--power-interval-ms", type=int, default=100, help="tegrastats sampling interval")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU provider for ONNX")
    return parser.parse_args()


def list_images(sample_dir: Path) -> List[Path]:
    return sorted([p for p in sample_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])


def preprocess_image(image_path: Path, input_size: int) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    return image


def make_batches(chw_images: Sequence[np.ndarray], batch_size: int) -> List[np.ndarray]:
    if not chw_images:
        raise ValueError("No images available to form batches.")
    valid_count = (len(chw_images) // batch_size) * batch_size
    if valid_count == 0:
        first = chw_images[0]
        return [np.stack([first for _ in range(batch_size)], axis=0).astype(np.float32)]
    batches = []
    for i in range(0, valid_count, batch_size):
        batches.append(np.stack(chw_images[i:i + batch_size], axis=0).astype(np.float32))
    return batches


class OnnxBackend:
    def __init__(self, model_path: Path, cpu_only: bool = False):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for --model-type onnx") from exc

        providers = ["CPUExecutionProvider"] if cpu_only else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, batch_nchw: np.ndarray) -> np.ndarray:
        outputs = self.session.run(None, {self.input_name: batch_nchw})
        return outputs[0] if isinstance(outputs, list) else outputs


class TrtBackend:
    def __init__(self, model_path: Path, input_shape: Tuple[int, int, int, int]):
        try:
            import tensorrt as trt
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
        except ImportError as exc:
            raise RuntimeError("TensorRT Python and pycuda are required for --model-type engine") from exc

        self.trt = trt
        self.cuda = cuda
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        engine_data = model_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {model_path}")

        self.context = self.engine.create_execution_context()
        self.input_index = next(i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i))
        self.output_index = next(i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i))
        self.bindings = [None] * self.engine.num_bindings
        self.stream = cuda.Stream()

        self.host_input = None
        self.host_output = None
        self.device_input = None
        self.device_output = None
        self.active_input_shape = None
        self._allocate(input_shape)

    def _allocate(self, input_shape: Tuple[int, int, int, int]) -> None:
        self.context.set_binding_shape(self.input_index, input_shape)
        output_shape = tuple(self.context.get_binding_shape(self.output_index))
        output_dtype = self.trt.nptype(self.engine.get_binding_dtype(self.output_index))
        input_dtype = self.trt.nptype(self.engine.get_binding_dtype(self.input_index))

        self.host_input = np.empty(input_shape, dtype=input_dtype)
        self.host_output = np.empty(output_shape, dtype=output_dtype)
        self.device_input = self.cuda.mem_alloc(self.host_input.nbytes)
        self.device_output = self.cuda.mem_alloc(self.host_output.nbytes)

        self.bindings[self.input_index] = int(self.device_input)
        self.bindings[self.output_index] = int(self.device_output)
        self.active_input_shape = input_shape

    def infer(self, batch_nchw: np.ndarray) -> np.ndarray:
        current_shape = tuple(batch_nchw.shape)
        if current_shape != self.active_input_shape:
            self._allocate(current_shape)

        np.copyto(self.host_input, batch_nchw)
        self.cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        self.cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
        self.stream.synchronize()
        return self.host_output.copy()


def build_backend(args: argparse.Namespace):
    if args.model_type == "onnx":
        return OnnxBackend(args.model_path, cpu_only=args.cpu_only)
    input_shape = (args.batch_size, 3, args.input_size, args.input_size)
    return TrtBackend(args.model_path, input_shape=input_shape)


@dataclass
class PowerStats:
    avg_watts: float
    max_watts: float
    samples: int


class TegrastatsMonitor:
    def __init__(self, log_path: Path, interval_ms: int):
        self.log_path = log_path
        self.interval_ms = interval_ms
        self.process = None
        self.log_file = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        self.process = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.log_file is not None:
            self.log_file.close()

    @staticmethod
    def parse(log_path: Path) -> PowerStats:
        if not log_path.exists():
            return PowerStats(avg_watts=0.0, max_watts=0.0, samples=0)

        pattern = re.compile(r"(?:VDD_IN|POM_5V_IN)\s+(\d+)mW")
        values = []
        with open(log_path, "r", encoding="utf-8") as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    values.append(float(match.group(1)) / 1000.0)

        if not values:
            return PowerStats(avg_watts=0.0, max_watts=0.0, samples=0)

        return PowerStats(
            avg_watts=float(np.mean(values)),
            max_watts=float(np.max(values)),
            samples=len(values),
        )


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float32), p))


def benchmark(args: argparse.Namespace) -> dict:
    images = list_images(args.sample_dir)
    if not images:
        raise ValueError(f"No images found under: {args.sample_dir}")

    chw_images = [preprocess_image(path, args.input_size) for path in images]
    batches = make_batches(chw_images, args.batch_size)
    backend = build_backend(args)

    warmup_batch = batches[0]
    for _ in range(args.warmup):
        _ = backend.infer(warmup_batch)

    model_name = args.model_path.stem
    run_dir = args.output_dir / model_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    power_log_path = run_dir / "tegrastats.log"

    monitor = TegrastatsMonitor(power_log_path, interval_ms=args.power_interval_ms)
    monitor.start()

    timings_ms = []
    batch_count = len(batches)
    images_processed = 0
    start_total = time.perf_counter()

    for i in range(args.runs):
        batch = batches[i % batch_count]
        start = time.perf_counter()
        _ = backend.infer(batch)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings_ms.append(elapsed_ms)
        images_processed += batch.shape[0]

    total_seconds = time.perf_counter() - start_total
    monitor.stop()
    power = TegrastatsMonitor.parse(power_log_path)

    mean_batch_ms = float(np.mean(timings_ms)) if timings_ms else 0.0
    mean_image_ms = mean_batch_ms / args.batch_size if args.batch_size > 0 else 0.0
    throughput = images_processed / total_seconds if total_seconds > 0 else 0.0
    fps = 1000.0 / mean_image_ms if mean_image_ms > 0 else 0.0
    energy_per_image_j = power.avg_watts / throughput if throughput > 0 else 0.0

    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(args.model_path),
        "model_type": args.model_type,
        "sample_dir": str(args.sample_dir),
        "batch_size": args.batch_size,
        "input_size": args.input_size,
        "warmup": args.warmup,
        "runs": args.runs,
        "images_seen": len(images),
        "images_processed": images_processed,
        "latency_ms": {
            "mean_batch": mean_batch_ms,
            "mean_image": mean_image_ms,
            "p50_batch": percentile(timings_ms, 50),
            "p95_batch": percentile(timings_ms, 95),
        },
        "throughput_images_per_sec": throughput,
        "fps": fps,
        "power_watts": {
            "avg": power.avg_watts,
            "max": power.max_watts,
            "samples": power.samples,
        },
        "energy_per_image_j": energy_per_image_j,
        "artifacts": {
            "power_log": str(power_log_path),
        },
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with open(run_dir / "metrics.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["latency_mean_batch_ms", metrics["latency_ms"]["mean_batch"]])
        writer.writerow(["latency_mean_image_ms", metrics["latency_ms"]["mean_image"]])
        writer.writerow(["latency_p50_batch_ms", metrics["latency_ms"]["p50_batch"]])
        writer.writerow(["latency_p95_batch_ms", metrics["latency_ms"]["p95_batch"]])
        writer.writerow(["throughput_images_per_sec", metrics["throughput_images_per_sec"]])
        writer.writerow(["fps", metrics["fps"]])
        writer.writerow(["power_avg_watts", metrics["power_watts"]["avg"]])
        writer.writerow(["power_max_watts", metrics["power_watts"]["max"]])
        writer.writerow(["energy_per_image_j", metrics["energy_per_image_j"]])

    return {"metrics": metrics, "run_dir": run_dir}


def print_summary(result: dict) -> None:
    metrics = result["metrics"]
    print("=" * 72)
    print(f"Model: {metrics['model_path']}")
    print(f"Type: {metrics['model_type']}")
    print("-" * 72)
    print(f"Latency (mean, image): {metrics['latency_ms']['mean_image']:.3f} ms")
    print(f"Latency (p95, batch):  {metrics['latency_ms']['p95_batch']:.3f} ms")
    print(f"Throughput:            {metrics['throughput_images_per_sec']:.3f} images/s")
    print(f"FPS:                   {metrics['fps']:.3f}")
    print(f"Power avg/max:         {metrics['power_watts']['avg']:.3f}/{metrics['power_watts']['max']:.3f} W")
    print(f"Energy/Image:          {metrics['energy_per_image_j']:.4f} J")
    print("-" * 72)
    print(f"Saved to: {result['run_dir']}")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {args.sample_dir}")

    result = benchmark(args)
    print_summary(result)


if __name__ == "__main__":
    main()