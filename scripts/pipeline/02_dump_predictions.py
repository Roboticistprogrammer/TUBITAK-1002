"""Dump per-image logits of ONE model on the held-out split, for every domain.

Works for a PyTorch checkpoint (student, no-KD student, or teacher), an ONNX file, or a TensorRT
engine, so the same file format feeds `08_compute_statistics.py` whichever machine produced it.
Run teachers with --checkpoint on the main machine; run the engine on the Jetson with --engine.

    python scripts/pipeline/02_dump_predictions.py --name kd_uniform_seed42 \
        --checkpoint outputs/retrain/kd_uniform/seed42/best.pt --dataset-root /data/fasdd
    python scripts/pipeline/02_dump_predictions.py --name kd_uniform_seed42_trt_fp16 \
        --engine artifacts/engines/kd_uniform_seed42_fp16.engine --dataset-root /data/fasdd

Output: <out-root>/<name>/<domain>.npz (logits, labels, names, classes) + <out-root>/<name>/meta.json.

External validation (external/USTC_PROTOCOL.md): add --dataset ustc to run the same artifact on data_index/ustc_smokers.csv
instead of the FASDD domains (add --preprocessing full_resize and a separate --out-root for the secondary
preprocessing of the protocol). It writes <out-root>/<name>/ustc.npz (logits, labels6, labels_bin, names, classes,
classes6) and meta.json. A torchvision CNN checkpoint (ResNet-50, MobileNetV3) needs --family cnn.

    python scripts/pipeline/02_dump_predictions.py --dataset ustc --name teacher_rs \
        --checkpoint outputs/teachers/FASDD_RS/best.pt --limit 50

Naming convention used by the statistics script: <group>_seed<N>[_<suffix>], e.g. kd_uniform_seed42,
kd_routed_seed43, nokd_seed44; teachers as teacher_cv / teacher_rs / teacher_uav (no seed).
"""
from __future__ import annotations

import argparse
import json
import subprocess
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
from firecls.data.ustc import USTC_CLASSES, USTC_INDEX, UstcDataset  # noqa: E402
from firecls.deployment.runners import OnnxRuntimeRunner, PyTorchRunner, TensorRTRunner  # noqa: E402
from firecls.transforms import build_eval_transform, build_full_resize_transform  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--onnx", type=Path)
    source.add_argument("--engine", type=Path)
    parser.add_argument("--family", default=None, help="baseline family of --checkpoint (e.g. cnn); default: SwinV2")
    parser.add_argument("--dataset", choices=["fasdd", "ustc"], default="fasdd")
    parser.add_argument("--ustc-dir", type=Path, default=None,
                        help="USTC only: folder holding <class>/<file>.tif, e.g. test-dataset/ustc_smokers; "
                             "default is datasets/USTC_SmokeRS beneath --dataset-root")
    parser.add_argument("--preprocessing", choices=["primary", "full_resize"], default="primary",
                        help="USTC only. primary = resize 224 + centre crop 192 (the FASDD transform); "
                             "full_resize = whole image resized to 192x192 (secondary sensitivity analysis). "
                             "Write each setting to its own --out-root.")
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
    if args.checkpoint and args.family:
        from firecls.baselines.registry import load_baseline

        loaded = load_baseline(args.family, args.checkpoint, device)
        spec = loaded.preprocessing
        if spec["mode"] != "resize_center_crop":
            raise ValueError(f"{args.family}: only resize_center_crop preprocessing is supported here")
        return "pytorch", PyTorchRunner(loaded.model, device), args.checkpoint, list(spec["mean"]), list(spec["std"])
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


def dump_ustc(args: argparse.Namespace, runner, transform, device: torch.device, out_dir: Path) -> dict[str, int]:
    """Logits on every USTC_SmokeRS image (all are `test`); the binary rule is applied later by evaluate_external.py."""
    index = args.index_root / USTC_INDEX
    if not index.exists():
        raise FileNotFoundError(f"{index} missing; run scripts/build_ustc_index.py first.")
    dataset = UstcDataset(index, args.dataset_root, transform, images_dir=args.ustc_dir)
    if len(dataset) == 0:
        raise ValueError(f"No test rows in {index}")
    if args.limit:  # evenly spaced, so a smoke test sees every class instead of only the first one
        step = max(1, len(dataset) // args.limit)
        dataset = Subset(dataset, range(0, len(dataset), step)[: args.limit])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    logits, labels6, labels_bin, names = [], [], [], []
    for step, (images, batch6, batch_bin, batch_paths) in enumerate(loader, 1):
        batch_logits = runner.infer(images)
        if batch_logits.ndim != 2 or batch_logits.shape[1] != len(CLASSES):
            raise ValueError(f"expected logits [batch,{len(CLASSES)}], got {batch_logits.shape}")
        logits.append(batch_logits)
        labels6.extend(batch6.tolist())
        labels_bin.extend(batch_bin.tolist())
        names.extend("/".join(Path(p.replace("\\", "/")).parts[-2:]) for p in batch_paths)  # class folder / file
        if step % 50 == 0:
            print(f"[ustc] {step * args.batch_size}/{len(dataset)}", flush=True)
    logits = np.concatenate(logits).astype(np.float32)
    np.savez_compressed(
        out_dir / "ustc.npz", logits=logits, labels6=np.asarray(labels6, dtype=np.int64),
        labels_bin=np.asarray(labels_bin, dtype=np.int64), names=np.asarray(names),
        classes=np.asarray(CLASSES), classes6=np.asarray(USTC_CLASSES),
    )
    predicted = np.bincount(logits.argmax(1), minlength=len(CLASSES))
    print(f"[ustc] saved {len(labels6)} predictions; predicted-class counts "
          f"{dict(zip(CLASSES, predicted.tolist()))}")
    if len(labels6) >= 20 and predicted.max() >= 0.95 * len(labels6):
        print("WARNING: >=95% of predictions are one class; check preprocessing before going on (external/README.md, Phase 3).")
    return {"ustc": len(labels6)}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kind, runner, artifact, mean, std = build_runner(args, device)
    if args.preprocessing != "primary" and args.dataset != "ustc":
        raise SystemExit("--preprocessing full_resize is only defined for --dataset ustc")
    build_transform = build_full_resize_transform if args.preprocessing == "full_resize" else build_eval_transform
    transform = build_transform(args.img_size, mean, std)

    out_dir = args.out_root / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    started = time.time()
    if args.dataset == "ustc":
        datasets = {}
        counts = dump_ustc(args, runner, transform, device, out_dir)
    else:
        datasets = build_domain_datasets(
            args.split, transform, args.dataset_root, args.domains, return_path=True, index_root=args.index_root
        )
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
        "name": args.name, "runtime": kind, "dataset": args.dataset, "preprocessing": args.preprocessing, "family": args.family,
        "split": "test" if args.dataset == "ustc" else args.split, "classes": CLASSES, "counts": counts,
        "artifact": str(artifact), "artifact_sha256": sha256_file(artifact), "img_size": args.img_size,
        "mean": mean, "std": std, "limit": args.limit, "seconds": round(time.time() - started, 1),
        "created": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(ROOT),
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    try:  # tracked files that differ from git_commit; a real run should have this empty
        meta["git_dirty_files"] = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        meta["git_dirty_files"] = None
    if args.dataset == "ustc":
        meta["index_sha256"] = sha256_file(args.index_root / USTC_INDEX)
    if kind == "onnxruntime":
        meta["providers"] = runner.providers
    write_json(out_dir / "meta.json", meta)
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
