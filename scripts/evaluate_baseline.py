"""Evaluate one baseline (PyTorch, ONNX and/or TensorRT) under the shared protocol.

Output schema matches ``scripts/evaluate_deployments.py`` so the student and every baseline can
be merged into one table by ``scripts/collect_results_table.py``.

Example:
    python scripts/evaluate_baseline.py --family cnn --name resnet50_seed42 \
        --checkpoint outputs/baselines/resnet50/seed42/best.pt \
        --onnx artifacts/baselines/resnet50_seed42.onnx \
        --engine artifacts/baselines/resnet50_seed42_fp16.engine \
        --output results/baselines/resnet50_seed42.json
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from bootstrap import setup_path

ROOT = setup_path()

from firecls.baselines.data import domain_loaders
from firecls.baselines.preprocessing import build_transform
from firecls.baselines.protocol import CLASSES, EVAL_SPLIT, SOURCES, protocol_dict
from firecls.baselines.registry import available_families, load_baseline
from firecls.deployment.artifacts import git_commit, sha256_file, write_json
from firecls.deployment.runners import OnnxRuntimeRunner, PyTorchRunner, TensorRTRunner
from firecls.evaluation import classification_metrics, prediction_agreement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", required=True, help=f"one of {available_families()}")
    parser.add_argument("--name", required=True, help="row name, e.g. resnet50_seed42")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--source", choices=SOURCES, default=None, help="provenance marker; default from manifest")
    parser.add_argument("--dataset-root", type=Path, default=ROOT, help="directory that contains datasets/")
    parser.add_argument("--index-root", type=Path, default=ROOT, help="directory that contains data_index/")
    parser.add_argument("--split", choices=["val", "test"], default=EVAL_SPLIT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def check_manifest(artifact: Path, checkpoint: Path, preprocessing: dict) -> dict:
    path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    if not path.exists():
        raise FileNotFoundError(f"Manifest required next to {artifact}: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("classes") != CLASSES:
        raise ValueError(f"{path}: class order {manifest.get('classes')} != {CLASSES}")
    source_hash = manifest.get("source_checkpoint_sha256")
    if source_hash and source_hash != sha256_file(checkpoint):
        raise ValueError(f"{artifact} was not exported from {checkpoint}")
    if manifest.get("preprocessing", preprocessing) != preprocessing:
        raise ValueError(f"{path}: preprocessing differs from the PyTorch checkpoint's")
    return manifest


def run_domain(runner, loader) -> tuple[dict, np.ndarray, np.ndarray, list[str]]:
    labels, paths, batches = [], [], []
    for images, batch_labels, batch_paths in loader:
        scores = runner.infer(images)
        if scores.ndim != 2 or scores.shape[1] != len(CLASSES):
            raise ValueError(f"Expected [batch, {len(CLASSES)}] scores, got {scores.shape}")
        batches.append(scores)
        labels.extend(batch_labels.tolist())
        paths.extend(batch_paths)
    scores = np.concatenate(batches)
    labels = np.asarray(labels, dtype=np.int64)
    return classification_metrics(labels, scores.argmax(1), CLASSES), labels, scores, paths


def macro(domains: dict) -> dict:
    names = ["accuracy", "balanced_accuracy", "macro_f1_present_classes"]
    return {n: float(np.mean([d["metrics"][n] for d in domains.values()])) for n in names}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_baseline(args.family, args.checkpoint, device)
    preprocessing = loaded.preprocessing
    loaders = domain_loaders(
        args.split, build_transform(preprocessing), args.batch_size, args.num_workers, args.dataset_root,
        return_path=True, index_root=args.index_root,
    )
    source = args.source or loaded.metadata.get("source", "official")

    results = {
        "protocol": {**protocol_dict(), "split": args.split, "preprocessing": preprocessing, "git_commit": git_commit(ROOT)},
        "baseline": {"name": args.name, "family": args.family, "source": source, **{
            k: v for k, v in loaded.metadata.items() if isinstance(v, (str, int, float, bool, list, dict)) or v is None
        }},
        "models": {},
        "runtime_agreement": {},
    }
    captured: dict[str, dict] = {}

    def evaluate(runtime: str, runner, artifact: Path) -> None:
        domains, capture = {}, {}
        for domain, loader in loaders.items():
            metrics, labels, scores, paths = run_domain(runner, loader)
            domains[domain] = {
                "metrics": metrics,
                "ordered_path_sha256": hashlib.sha256("\n".join(paths).encode()).hexdigest(),
            }
            capture[domain] = (labels, scores, paths)
            print(f"{args.name}/{runtime}/{domain}: acc={metrics['accuracy']:.4f} "
                  f"macro_f1={metrics['macro_f1_present_classes']:.4f} n={metrics['samples']}")
        results["models"][f"{args.name}_{runtime}"] = {
            "family": args.family,
            "runtime": runtime,
            "artifact": str(artifact),
            "artifact_sha256": sha256_file(artifact),
            "domains": domains,
            "domain_macro": macro(domains),
        }
        captured[runtime] = capture

    evaluate("pt", PyTorchRunner(loaded.model, device), args.checkpoint)
    del loaded
    gc.collect()
    if args.onnx:
        check_manifest(args.onnx, args.checkpoint, preprocessing)
        evaluate("onnx", OnnxRuntimeRunner(args.onnx, use_cuda=device.type == "cuda"), args.onnx)
    if args.engine:
        check_manifest(args.engine, args.checkpoint, preprocessing)
        evaluate("engine", TensorRTRunner(args.engine, device), args.engine)

    reference = captured["pt"]
    for runtime, capture in captured.items():
        if runtime == "pt":
            continue
        per_domain = {}
        for domain in reference:
            ref_labels, ref_scores, ref_paths = reference[domain]
            labels, scores, paths = capture[domain]
            if ref_paths != paths or not np.array_equal(ref_labels, labels):
                raise RuntimeError(f"Sample order changed for {runtime}/{domain}")
            per_domain[domain] = prediction_agreement(ref_scores, scores)
        results["runtime_agreement"][runtime] = per_domain

    write_json(args.output, results)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
