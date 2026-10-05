"""Create a TensorRT INT8 calibration cache from a folder of TRAINING images (never test images).

    python scripts/pipeline/04a_make_int8_calibration.py --onnx artifacts/retrain/kd_uniform_seed42.onnx \
        --images data/calib --output artifacts/calibration.cache

Use 500-1000 images, stratified by domain and class. Runs on the Jetson (TensorRT 10.x, entropy calibrator 2).
The cache is then passed to 04_build_engines.sh via CALIB_CACHE. INT8 accuracy must be re-checked with
02_dump_predictions.py --engine on the INT8 engine (issue 2.8); INT8 latency alone says nothing about accuracy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import setup_path  # noqa: E402

setup_path()

from firecls.config import DEFAULT_IMG_SIZE  # noqa: E402
from firecls.transforms import build_eval_transform  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("dump", Path(__file__).resolve().parent / "02_dump_predictions.py")
_dump = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dump)
preprocessing_from_manifest = _dump.preprocessing_from_manifest

SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/calibration.cache"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--workspace-mib", type=int, default=2048)
    args = parser.parse_args()

    import tensorrt as trt

    files = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in SUFFIXES)[: args.limit]
    if len(files) < args.batch_size:
        raise SystemExit(f"need at least {args.batch_size} images under {args.images}, found {len(files)}")
    mean, std = preprocessing_from_manifest(args.onnx)
    transform = build_eval_transform(DEFAULT_IMG_SIZE, mean, std)
    batch_size = args.batch_size

    class Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.index = 0
            self.buffer = torch.empty(batch_size, 3, DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE, device="cuda")

        def get_batch_size(self):
            return batch_size

        def get_batch(self, names):
            if self.index + batch_size > len(files):
                return None
            batch = torch.stack([transform(Image.open(f).convert("RGB")) for f in files[self.index : self.index + batch_size]])
            self.index += batch_size
            self.buffer.copy_(batch)
            return [int(self.buffer.data_ptr())]

        def read_calibration_cache(self):
            return None

        def write_calibration_cache(self, cache):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(cache)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # explicit batch is the default in TensorRT 10
    parser_onnx = trt.OnnxParser(network, logger)
    if not parser_onnx.parse(args.onnx.read_bytes()):
        raise SystemExit("\n".join(str(parser_onnx.get_error(i)) for i in range(parser_onnx.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 1024**2)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    shape = (batch_size, 3, DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE)
    profile = builder.create_optimization_profile()
    profile.set_shape(network.get_input(0).name, shape, shape, shape)
    config.add_optimization_profile(profile)
    config.set_calibration_profile(profile)
    config.int8_calibrator = Calibrator()
    if builder.build_serialized_network(network, config) is None:
        raise SystemExit("calibration build failed")
    print(f"Calibration cache written: {args.output} ({len(files)} images, batch {batch_size})")


if __name__ == "__main__":
    main()
