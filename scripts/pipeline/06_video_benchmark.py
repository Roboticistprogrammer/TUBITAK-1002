"""End-to-end video benchmark with a per-stage latency breakdown (issue 2.9).

Stages, timed per frame: decode (OpenCV), preprocess (BGR->RGB, resize 224, center-crop 192, normalise: the
same transform used for the accuracy evaluation), inference (runner.infer = H2D + compute + D2H) and
postprocess (softmax + argmax). Reports mean/p50/p95/p99 of each stage, end-to-end FPS, inference-only FPS and the
time not explained by the four stages (loop and Python overhead). With a batch size above 1 the batch-level stage
times are divided by the batch size.

    python scripts/pipeline/06_video_benchmark.py --name kd_uniform_seed42_trt_fp16 \
        --engine artifacts/engines/kd_uniform_seed42_fp16.engine --videos videos/flame2_fhd.mp4 --batch-size 1

Run one runtime per process. For FLAME 2 pass the FHD and 4K clips; frames are decoded at their native size and
resized by the transform, so 4K mainly stresses the decode and preprocess stages. Run 03_jetson_env.sh first.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import setup_path  # noqa: E402

ROOT = setup_path()

from firecls.baselines.power import TegrastatsMonitor, system_ram_used_mb  # noqa: E402
from firecls.config import DEFAULT_IMG_SIZE  # noqa: E402
from firecls.deployment.artifacts import git_commit, sha256_file, write_json  # noqa: E402
from firecls.deployment.runners import OnnxRuntimeRunner, PyTorchRunner, TensorRTRunner  # noqa: E402
from firecls.transforms import build_eval_transform  # noqa: E402

_spec = importlib.util.spec_from_file_location("dump", Path(__file__).resolve().parent / "02_dump_predictions.py")
_dump = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dump)


def stats(values_ms: list[float]) -> dict:
    arr = np.asarray(values_ms)
    return {"mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)), "p99_ms": float(np.percentile(arr, 99)), "n": int(arr.size)}


def softmax_argmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.argmax(axis=1)


def cv2_transform(mean, std):
    """Faster CPU preprocessing (OpenCV bilinear resize to 224, center-crop 192); not bit-identical to the PIL
    transform used for the accuracy evaluation, so report it as a separate pipeline variant."""
    import cv2

    mean_arr, std_arr = np.asarray(mean, np.float32), np.asarray(std, np.float32)

    def apply(frame_bgr):
        height, width = frame_bgr.shape[:2]
        scale = (DEFAULT_IMG_SIZE + 32) / min(height, width)
        resized = cv2.resize(frame_bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_LINEAR)
        top, left = (resized.shape[0] - DEFAULT_IMG_SIZE) // 2, (resized.shape[1] - DEFAULT_IMG_SIZE) // 2
        crop = resized[top : top + DEFAULT_IMG_SIZE, left : left + DEFAULT_IMG_SIZE, ::-1].astype(np.float32) / 255.0
        return torch.from_numpy(((crop - mean_arr) / std_arr).transpose(2, 0, 1).copy())

    return apply


def run_video(video: Path, runner, transform, batch_size: int, warmup_frames: int, max_frames: int | None, use_cv2: bool = False) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {video}")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_source = capture.get(cv2.CAP_PROP_FPS)
    stages = {"decode": [], "preprocess": [], "inference": [], "postprocess": []}
    frames = measured = 0
    pending: list[torch.Tensor] = []
    wall_start = None
    while True:
        started = time.perf_counter()
        ok, frame = capture.read()
        decode_ms = (time.perf_counter() - started) * 1000.0
        if not ok or (max_frames is not None and frames >= max_frames + warmup_frames):
            break
        started = time.perf_counter()
        tensor = transform(frame) if use_cv2 else transform(Image.fromarray(frame[:, :, ::-1]))
        preprocess_ms = (time.perf_counter() - started) * 1000.0
        frames += 1
        pending.append(tensor)
        record = frames > warmup_frames
        if record:
            stages["decode"].append(decode_ms)
            stages["preprocess"].append(preprocess_ms)
            wall_start = wall_start or time.perf_counter() - (decode_ms + preprocess_ms) / 1000.0
        if len(pending) < batch_size:
            continue
        batch, pending = torch.stack(pending), []
        started = time.perf_counter()
        logits = runner.infer(batch)
        inference_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        softmax_argmax(logits)
        post_ms = (time.perf_counter() - started) * 1000.0
        if record:
            stages["inference"] += [inference_ms / batch_size] * batch_size
            stages["postprocess"] += [post_ms / batch_size] * batch_size
            measured += batch_size
    wall = time.perf_counter() - wall_start if wall_start else float("nan")
    capture.release()
    summary = {s: stats(v) for s, v in stages.items() if v}
    per_frame_sum = sum(summary[s]["mean_ms"] for s in summary)
    end_to_end_ms = 1000.0 * wall / max(1, measured)
    return {
        "video": str(video), "resolution": [width, height], "source_fps": fps_source,
        "warmup_frames": warmup_frames, "measured_frames": measured, "batch_size": batch_size,
        "stages": summary, "sum_of_stage_means_ms": per_frame_sum,
        "end_to_end_ms_per_frame": end_to_end_ms, "end_to_end_fps": 1000.0 / end_to_end_ms,
        "inference_only_fps": 1000.0 / summary["inference"]["mean_ms"] if "inference" in summary else None,
        "unexplained_ms_per_frame": end_to_end_ms - per_frame_sum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--onnx", type=Path)
    source.add_argument("--engine", type=Path)
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=None, help="measured frames per video (default: whole video)")
    parser.add_argument("--preprocess", choices=["pil", "cv2"], default="pil",
                        help="pil = same transform as the accuracy evaluation; cv2 = faster CPU variant")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    idle_ram = system_ram_used_mb()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kind, runner, artifact, mean, std = _dump.build_runner(args, device)
    transform = cv2_transform(mean, std) if args.preprocess == "cv2" else build_eval_transform(DEFAULT_IMG_SIZE, mean, std)

    results = []
    with TegrastatsMonitor(log_path=Path(f"results/jetson/{args.name}_video_tegrastats.log") if Path("results/jetson").exists() else None) as monitor:
        for video in args.videos:
            print(f"{args.name}: {video}", flush=True)
            results.append(run_video(video, runner, transform, args.batch_size, args.warmup_frames, args.max_frames, args.preprocess == "cv2"))
            print(f"  {results[-1]['end_to_end_fps']:.1f} FPS end-to-end, {results[-1]['inference_only_fps']:.1f} FPS inference only")
    payload = {
        "name": args.name, "runtime": kind, "artifact": str(artifact), "artifact_sha256": sha256_file(artifact),
        "git_commit": git_commit(ROOT), "idle_ram_used_mb": idle_ram, "videos": results,
        "power": monitor.summary(), "memory": monitor.memory_summary(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "preprocess": args.preprocess,
        "timing_scope": "per-frame stage means; inference = runtime infer call incl. H2D/D2H",
    }
    output = args.output or Path(f"results/jetson/{args.name}_video.json")
    write_json(output, payload)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
