"""
Benchmark model inference performance on host machine.
Measures: latency (ms), FPS, power draw (W), model weights (MB).
Supports: .pt, .onnx, .engine

Power draw requires pynvml (pip install pynvml). Falls back gracefully if unavailable.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from ultralytics import YOLO

# Optional GPU power monitoring via nvidia-ml-py (pip install nvidia-ml-py)
import warnings as _warnings
try:
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", FutureWarning)
        import pynvml  # nvidia-ml-py exposes the same pynvml API
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


def _init_nvml_handle() -> Optional[object]:
    """Initialize pynvml and return GPU handle for device 0, or None if unavailable."""
    if not _PYNVML_AVAILABLE:
        return None
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


def _read_power_watts(handle) -> Optional[float]:
    """Read instantaneous GPU power draw in watts. Returns None on failure."""
    if handle is None:
        return None
    try:
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
    except Exception:
        return None


def _shutdown_nvml() -> None:
    if _PYNVML_AVAILABLE:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def validate_pt_for_ultralytics(model_path: Path) -> str:
    if model_path.suffix.lower() != ".pt":
        return ""

    try:
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception as error:
        return f"Failed to read checkpoint: {error}"

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model_obj = checkpoint["model"]
        if isinstance(model_obj, dict):
            return (
                "This .pt appears to be a raw state_dict checkpoint (OrderedDict), not an Ultralytics YOLO model. "
                "Use your original training code to benchmark it, or export a compatible .onnx/.engine model."
            )

    return ""


def _timed_loop(run_fn, num_runs: int, nvml_handle) -> tuple:
    """Run run_fn num_runs times, return (timings_ms list, power_readings_w list)."""
    timings_ms, power_w = [], []
    for _ in range(num_runs):
        pw = _read_power_watts(nvml_handle)
        if pw is not None:
            power_w.append(pw)
        t0 = time.perf_counter()
        run_fn()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
        pw = _read_power_watts(nvml_handle)
        if pw is not None:
            power_w.append(pw)
    return timings_ms, power_w


def _build_batch_result(timings_ms, power_w, batch_size) -> Dict:
    avg_ms      = float(np.mean(timings_ms))
    std_ms      = float(np.std(timings_ms))
    per_img_ms  = avg_ms / max(batch_size, 1)
    fps         = float(1000.0 / per_img_ms) if per_img_ms > 0 else 0.0
    avg_pw      = float(np.mean(power_w)) if power_w else None
    r = {"latency_ms": per_img_ms, "std_ms": std_ms, "fps": fps,
         "throughput": fps, "batch_latency_ms": avg_ms}
    if avg_pw is not None:
        r["power_draw_w"] = avg_pw
    print(f"  Latency/image : {per_img_ms:.2f} ms  (±{std_ms:.2f})")
    print(f"  FPS           : {fps:.1f}")
    if avg_pw is not None:
        print(f"  Power draw    : {avg_pw:.1f} W")
    return r


def _benchmark_onnxruntime(model_path: Path, batch_sizes, imgsz, num_runs, nvml_handle) -> Dict:
    """Benchmark ONNX model using ONNX Runtime directly (avoids Ultralytics task guessing)."""
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(model_path), providers=providers)
    inp_name = sess.get_inputs()[0].name
    # Use the model's own declared static shape if available, else fall back to imgsz
    declared = sess.get_inputs()[0].shape
    h = declared[2] if isinstance(declared[2], int) and declared[2] > 0 else imgsz
    w = declared[3] if isinstance(declared[3], int) and declared[3] > 0 else imgsz
    print(f"  ONNX Runtime  : {ort.__version__}  |  input {h}×{w}")
    print(f"  Provider      : {sess.get_providers()[0]}")

    results: Dict = {}
    for bs in batch_sizes:
        print(f"\nBenchmarking batch_size={bs}")
        dummy = np.random.rand(bs, 3, h, w).astype(np.float32)
        feed  = {inp_name: dummy}
        for _ in range(10):            # warmup
            sess.run(None, feed)
        timings_ms, power_w = _timed_loop(lambda: sess.run(None, feed), num_runs, nvml_handle)
        results[f"batch_{bs}"] = _build_batch_result(timings_ms, power_w, bs)

    results["model_size_mb"] = float(model_path.stat().st_size / (1024 ** 2))
    _shutdown_nvml()
    return results


def _benchmark_tensorrt(model_path: Path, batch_sizes, imgsz, num_runs, nvml_handle) -> Dict:
    """Benchmark TensorRT .engine using tensorrt + pycuda directly."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except ImportError as exc:
        print(f"  [WARN] TensorRT/PyCUDA not available: {exc}. Reporting file size only.")
        _shutdown_nvml()
        return {
            "model_size_mb": float(model_path.stat().st_size / (1024 ** 2)),
            "batch_1": {"latency_ms": 0.0, "std_ms": 0.0, "fps": 0.0, "throughput": 0.0},
        }

    logger = trt.Logger(trt.Logger.WARNING)
    with open(model_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print(f"  [WARN] TensorRT could not deserialize engine.")
        print(f"         Installed TensorRT: {trt.__version__}")
        print(f"         The engine was likely built with a different TensorRT version.")
        print(f"         Reporting file size only. Re-export the engine with TRT {trt.__version__} to benchmark.")
        _shutdown_nvml()
        return {
            "model_size_mb": float(model_path.stat().st_size / (1024 ** 2)),
            "batch_1": {"latency_ms": 0.0, "std_ms": 0.0, "fps": 0.0, "throughput": 0.0},
            "error": f"Engine version mismatch (installed TRT {trt.__version__})",
        }
    # Allocate buffers for the first input/output binding
    input_idx  = engine.get_binding_index(engine.get_binding_name(0))
    output_idx = engine.get_binding_index(engine.get_binding_name(1))

    results: Dict = {}
    for bs in batch_sizes:
        print(f"\nBenchmarking batch_size={bs}")
        in_shape  = (bs, 3, imgsz, imgsz)
        out_shape = tuple(engine.get_binding_shape(output_idx))
        if out_shape[0] == -1:
            out_shape = (bs,) + out_shape[1:]

        h_in   = cuda.pagelocked_empty(int(np.prod(in_shape)),  dtype=np.float32)
        h_out  = cuda.pagelocked_empty(int(np.prod(out_shape)), dtype=np.float32)
        d_in   = cuda.mem_alloc(h_in.nbytes)
        d_out  = cuda.mem_alloc(h_out.nbytes)
        stream = cuda.Stream()

        np.copyto(h_in, np.random.rand(*in_shape).astype(np.float32).ravel())

        def _infer():
            cuda.memcpy_htod_async(d_in, h_in, stream)
            context.execute_async_v2(bindings=[int(d_in), int(d_out)], stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(h_out, d_out, stream)
            stream.synchronize()

        for _ in range(10):  # warmup
            _infer()

        timings_ms, power_w = _timed_loop(_infer, num_runs, nvml_handle)
        results[f"batch_{bs}"] = _build_batch_result(timings_ms, power_w, bs)

    results["model_size_mb"] = float(model_path.stat().st_size / (1024 ** 2))
    _shutdown_nvml()
    return results


def benchmark_model(model_path: Path, model_type: str, batch_sizes: list = [1], imgsz: int = 640, num_runs: int = 100) -> Dict:
    """Benchmark model: latency (ms), FPS, power draw (W), model weights (MB)."""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {model_path.name} ({model_type}) @ {imgsz}px")
    print(f"{'='*70}")

    nvml_handle = _init_nvml_handle()
    if nvml_handle is not None:
        print("  Power monitoring: enabled (pynvml)")
    else:
        print("  Power monitoring: disabled (install nvidia-ml-py for power draw)")

    # For ONNX / Engine use direct runtimes — Ultralytics guesses task=detect
    # which fails on classification models.
    if model_type in ("onnx",):
        return _benchmark_onnxruntime(model_path, batch_sizes, imgsz, num_runs, nvml_handle)
    if model_type in ("engine",):
        return _benchmark_tensorrt(model_path, batch_sizes, imgsz, num_runs, nvml_handle)

    # .pt via Ultralytics (detection / YOLO)
    model = YOLO(str(model_path))
    dummy_image = (np.random.rand(imgsz, imgsz, 3) * 255).astype(np.uint8)

    # Initialize GPU power handle once
    nvml_handle = _init_nvml_handle()
    if nvml_handle is not None:
        print("  Power monitoring: enabled (pynvml)")
    else:
        print("  Power monitoring: disabled (install pynvml for power draw)")

    results = {}
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch_size={batch_size}")
        source = [dummy_image.copy() for _ in range(batch_size)]

        # Warmup
        for _ in range(10):
            _ = model.predict(source=source, imgsz=imgsz, conf=0.25, iou=0.6, verbose=False)

        timings_ms = []
        power_readings_w = []

        for _ in range(num_runs):
            # Sample power before inference
            pw = _read_power_watts(nvml_handle)
            if pw is not None:
                power_readings_w.append(pw)

            start = time.perf_counter()
            _ = model.predict(source=source, imgsz=imgsz, conf=0.25, iou=0.6, verbose=False)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings_ms.append(elapsed_ms)

            # Sample power after inference
            pw = _read_power_watts(nvml_handle)
            if pw is not None:
                power_readings_w.append(pw)

        avg_ms = float(np.mean(timings_ms))
        std_ms = float(np.std(timings_ms))
        per_image_ms = avg_ms / max(batch_size, 1)
        fps = float(1000.0 / per_image_ms) if per_image_ms > 0 else 0.0
        avg_power_w = float(np.mean(power_readings_w)) if power_readings_w else None

        batch_result = {
            "latency_ms": per_image_ms,
            "std_ms": std_ms,
            "fps": fps,
            "throughput": fps,
            "batch_latency_ms": avg_ms,
        }
        if avg_power_w is not None:
            batch_result["power_draw_w"] = avg_power_w

        results[f"batch_{batch_size}"] = batch_result

        print(f"  Latency/image : {per_image_ms:.2f} ms  (±{std_ms:.2f})")
        print(f"  FPS           : {fps:.1f}")
        if avg_power_w is not None:
            print(f"  Power draw    : {avg_power_w:.1f} W")

    model_size_mb = model_path.stat().st_size / (1024 ** 2)
    results["model_size_mb"] = float(model_size_mb)

    print(f"\n{'='*70}")
    print(f"Model weights : {model_size_mb:.2f} MB")
    print(f"{'='*70}")

    _shutdown_nvml()
    return results


def benchmark_model_raw_pt(model_path: Path, num_runs: int = 50) -> Dict:
    """
    Fallback benchmark for non-Ultralytics .pt checkpoints.
    Extracts an nn.Module from the checkpoint if possible and times dummy
    inference. If no callable model is found, only file size is reported.
    """
    print(f"\n{'='*70}")
    print(f"Raw-PT Benchmarking: {model_path.name}")
    print(f"{'='*70}")

    results: Dict = {}
    model_size_mb = model_path.stat().st_size / (1024 ** 2)
    results["model_size_mb"] = float(model_size_mb)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nvml_handle = _init_nvml_handle()

    # Try to extract a callable nn.Module from the checkpoint
    model_obj = None
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, torch.nn.Module):
            model_obj = ckpt
        elif isinstance(ckpt, dict):
            for key in ("model", "net", "state_dict"):
                candidate = ckpt.get(key)
                if isinstance(candidate, torch.nn.Module):
                    model_obj = candidate
                    break
    except Exception as exc:
        print(f"  [WARN] Could not load checkpoint: {exc}")

    if model_obj is not None:
        model_obj = model_obj.to(device).eval()
        # Try common classification / transformer input shapes
        dummy = None
        for shape in ((1, 3, 192, 192), (1, 3, 224, 224), (1, 3, 384, 384)):
            try:
                with torch.no_grad():
                    _ = model_obj(torch.randn(*shape, device=device))
                dummy = torch.randn(*shape, device=device)
                break
            except Exception:
                continue

        if dummy is not None:
            print(f"  Running inference on {device} with input {tuple(dummy.shape)}")
            with torch.no_grad():                       # warmup
                for _ in range(10):
                    _ = model_obj(dummy)

            timings_ms, power_readings_w = [], []
            with torch.no_grad():
                for _ in range(num_runs):
                    pw = _read_power_watts(nvml_handle)
                    if pw is not None:
                        power_readings_w.append(pw)
                    t0 = time.perf_counter()
                    _ = model_obj(dummy)
                    timings_ms.append((time.perf_counter() - t0) * 1000.0)
                    pw = _read_power_watts(nvml_handle)
                    if pw is not None:
                        power_readings_w.append(pw)

            avg_ms  = float(np.mean(timings_ms))
            std_ms  = float(np.std(timings_ms))
            fps     = float(1000.0 / avg_ms) if avg_ms > 0 else 0.0
            avg_pw  = float(np.mean(power_readings_w)) if power_readings_w else None

            batch_result: Dict = {
                "latency_ms": avg_ms, "std_ms": std_ms,
                "fps": fps, "throughput": fps, "batch_latency_ms": avg_ms,
            }
            if avg_pw is not None:
                batch_result["power_draw_w"] = avg_pw
            results["batch_1"] = batch_result

            print(f"  Latency/image : {avg_ms:.2f} ms  (\u00b1{std_ms:.2f})")
            print(f"  FPS           : {fps:.1f}")
            if avg_pw is not None:
                print(f"  Power draw    : {avg_pw:.1f} W")
        else:
            print("  [WARN] No working input shape found — reporting file size only.")
            results["batch_1"] = {"latency_ms": 0.0, "std_ms": 0.0, "fps": 0.0, "throughput": 0.0}
    else:
        print("  [INFO] Checkpoint is a raw state_dict — reporting file size only.")
        results["batch_1"] = {"latency_ms": 0.0, "std_ms": 0.0, "fps": 0.0, "throughput": 0.0}

    print(f"\nModel weights : {model_size_mb:.2f} MB")
    print(f"{'='*70}")
    _shutdown_nvml()
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
        default=[1, 4, 8, 16],
        help="Batch sizes to benchmark (default: 1 4 8 16)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)"
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

    compatibility_error = validate_pt_for_ultralytics(args.model_path)
    if compatibility_error:
        print(f"[INFO] Not Ultralytics-compatible: {compatibility_error}")
        print("[INFO] Falling back to raw PyTorch benchmark...")
        results = benchmark_model_raw_pt(args.model_path, num_runs=args.num_runs)
    else:
        # Benchmark with Ultralytics
        results = benchmark_model(args.model_path, args.model_type, args.batch_sizes, imgsz=args.imgsz, num_runs=args.num_runs)
    
    # Latency measured at batch=1 (most fair per-image comparison)
    batch_1_results = results.get("batch_1", {})
    weights_mb = results.get("model_size_mb", 0.0)

    # Maximum throughput across all tested batch sizes
    max_throughput = max(
        (
            v.get("throughput", v.get("fps", 0.0))
            for k, v in results.items()
            if k.startswith("batch_") and isinstance(v, dict)
        ),
        default=0.0,
    )

    deployment = {
        "latency_ms"               : batch_1_results.get("latency_ms", 0.0),
        "fps"                      : batch_1_results.get("fps", 0.0),
        "throughput"               : batch_1_results.get("throughput", 0.0),
        "throughput_samples_per_sec": max_throughput,  # max across all batch sizes
        "weights_mb"               : weights_mb,
        "size_mb"                  : weights_mb,       # alias for backward compat
    }

    if "power_draw_w" in batch_1_results:
        deployment["power_draw_w"] = batch_1_results["power_draw_w"]

    output_data = {
        "model_name"       : args.model_path.stem,
        "model_type"       : args.model_type,
        "deployment"       : deployment,
        "all_batch_results": results,
    }

    # Save result
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
