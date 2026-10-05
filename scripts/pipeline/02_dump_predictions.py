"""Dump per-image logits of ONE model on the held-out split, for every domain.

Works for a PyTorch checkpoint (student, no-KD student, or teacher), an ONNX file, or a TensorRT
engine, so the same file format feeds `08_compute_statistics.py` whichever machine produced it.
Run teachers with --checkpoint on the main machine; run the engine on the Jetson with --engine.

    python scripts/pipeline/02_dump_predictions.py --name kd_uniform_seed42 \
        --checkpoint outputs/retrain/kd_uniform/seed42/best.pt --dataset-root /data/fasdd
    python scripts/pipeline/02_dump_predictions.py --name kd_uniform_seed42_trt_fp16 \
        --engine artifacts/engines/kd_uniform_seed42_fp16.engine --dataset-root /data/fasdd

Output: <out-root>/<name>/<domain>.npz (logits, labels, names, classes) + <out-root>/<name>/meta.json.
Naming convention used by the statistics script: <group>_seed<N>[_<suffix>], e.g. kd_uniform_seed42,
kd_routed_seed43, nokd_seed44; teachers as teacher_cv / teacher_rs / teacher_uav (no seed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import setup_path  # noqa: E402

ROOT = setup_path()

from firecls.baselines.data import build_domain_datasets  # noqa: E402
from firecls.baselines.preprocessing import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from firecls.baselines.protocol import CLASSES, DOMAINS  # noqa: E402
from firecls.config import DEFAULT_IMG_SIZE  # noqa: E402
from firecls.deployment.artifacts import git_commit, sha256_file, write_json  # noqa: E402
from firecls.deployment.runners import OnnxRuntimeRunner, PyTorchRunner, TensorRTRunner  # noqa: E402
from firecls.transforms import build_eval_transform  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--onnx", type=Path)
    source.add_argument("--engine", type=Path)
    parser.add_argument("--out-root", type=Path, default=Path("preds"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--dataset-root", type=Path, default=ROOT, help="directory containing datasets/")
    parser.add_argument("--index-root", type=Path, default=ROOT, help="directory containing data_index/")
    parser.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="first N images per domain (smoke test only)")
    return parser.parse_args()


def preprocessing_from_manifest(artifact: Path) -> tuple[list[float], list[float]]:
    """mean/std recorded by export_student.py; engines inherit them from their source ONNX manifest."""
    manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "preprocessing" not in manifest and manifest.get("source_onnx"):
            return preprocessing_from_manifest(Path(manifest["source_onnx"]))
        preprocessing = manifest.get("preprocessing")
        if preprocessing:
            return list(preprocessing["mean"]), list(preprocessing["std"])
    return list(IMAGENET_MEAN), list(IMAGENET_STD)


def build_runner(args: argparse.Namespace, device: torch.device):
    if args.checkpoint:
        from firecls.deployment.model import load_checkpoint_model

        model, processor, metadata = load_checkpoint_model(args.checkpoint, device)
        if list(metadata["classes"]) != list(CLASSES):
            raise ValueError(f"class order {metadata['classes']} != {CLASSES}")
        return "pytorch", PyTorchRunner(model, device), args.checkpoint, list(processor.image_mean), list(processor.image_std)
    if args.onnx:
        mean, std = preprocessing_from_manifest(args.onnx)
        return "onnxruntime", OnnxRuntimeRunner(args.onnx, use_cuda=device.type == "cuda"), args.onnx, mean, std
    mean, std = preprocessing_from_manifest(args.engine)
    return "tensorrt", TensorRTRunner(args.engine, device), args.engine, mean, std


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kind, runner, artifact, mean, std = build_runner(args, device)
    transform = build_eval_transform(args.img_size, mean, std)
    datasets = build_domain_datasets(
        args.split, transform, args.dataset_root, args.domains, return_path=True, index_root=args.index_root
    )

    out_dir = args.out_root / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    started = time.time()
    for domain, dataset in datasets.items():
        if args.limit:
            dataset = Subset(dataset, range(min(args.limit, len(dataset))))
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        logits, labels, names = [], [], []
        for step, (images, batch_labels, batch_paths) in enumerate(loader, 1):
            batch_logits = runner.infer(images)
            if batch_logits.ndim != 2 or batch_logits.shape[1] != len(CLASSES):
                raise ValueError(f"expected logits [batch,{len(CLASSES)}], got {batch_logits.shape}")
            logits.append(batch_logits)
            labels.extend(batch_labels.tolist())
            names.extend(Path(p.replace("\\", "/")).name for p in batch_paths)
            if step % 50 == 0:
                print(f"[{domain}] {step * args.batch_size}/{len(dataset)}", flush=True)
        logits = np.concatenate(logits).astype(np.float32)
        np.savez_compressed(
            out_dir / f"{domain}.npz",
            logits=logits, labels=np.asarray(labels, dtype=np.int64),
            names=np.asarray(names), classes=np.asarray(CLASSES),
        )
        counts[domain] = len(labels)
        print(f"[{domain}] saved {len(labels)} predictions, accuracy {(logits.argmax(1) == np.asarray(labels)).mean():.4f}")

    meta = {
        "name": args.name, "runtime": kind, "split": args.split, "classes": CLASSES, "counts": counts,
        "artifact": str(artifact), "artifact_sha256": sha256_file(artifact), "img_size": args.img_size,
        "mean": mean, "std": std, "limit": args.limit, "seconds": round(time.time() - started, 1),
        "created": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(ROOT),
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    if kind == "onnxruntime":
        meta["providers"] = runner.providers
    write_json(out_dir / "meta.json", meta)
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
