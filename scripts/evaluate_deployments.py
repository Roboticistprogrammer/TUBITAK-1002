from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bootstrap import setup_path

ROOT = setup_path()

from firecls.config import DEFAULT_IMG_SIZE, get_dataset_specs
from firecls.data.dataset import ImageClassificationCSVDataset
from firecls.deployment.artifacts import git_commit, sha256_file, write_json
from firecls.deployment.model import load_checkpoint_model
from firecls.deployment.runners import OnnxRuntimeRunner, PyTorchRunner, TensorRTRunner
from firecls.evaluation import classification_metrics, prediction_agreement
from firecls.transforms import build_eval_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate teachers and every student runtime on identical held-out samples."
    )
    parser.add_argument("--student-pt", type=Path, required=True)
    parser.add_argument("--student-onnx", type=Path)
    parser.add_argument("--student-engine", type=Path)
    parser.add_argument("--teacher-cv", type=Path)
    parser.add_argument("--teacher-rs", type=Path)
    parser.add_argument("--teacher-uav", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--output", type=Path, default=Path("results/classification_evaluation.json"))
    return parser.parse_args()


def validate_manifest(
    artifact: Path, classes: list[str], img_size: int, student_checkpoint: Path
) -> dict:
    path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    if not path.exists():
        raise FileNotFoundError(f"Artifact manifest is required: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("classes") != classes:
        raise ValueError(f"Class-order mismatch in {path}: {manifest.get('classes')} != {classes}")
    if manifest.get("input", {}).get("shape", [None, None, None, None])[-2:] != [img_size, img_size]:
        raise ValueError(f"Input-size mismatch in {path}; expected {img_size}x{img_size}.")
    source_hash = manifest.get("source_checkpoint_sha256")
    if source_hash and source_hash != sha256_file(student_checkpoint):
        raise ValueError(f"{artifact} was not exported from the supplied student checkpoint.")
    return manifest


def build_loader(spec, domain: str, split: str, classes, transform, args) -> DataLoader:
    index_csv = ROOT / "data_index" / f"{spec.name.lower()}_tdml.csv"
    dataset = ImageClassificationCSVDataset(
        index_csv,
        split,
        classes,
        transform=transform,
        return_path=True,
        images_root=spec.images_root,
    )
    if not dataset:
        raise ValueError(f"No {split} samples found for {domain} in {index_csv}")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_domain(runner, loader: DataLoader, classes: list[str]) -> tuple[dict, np.ndarray, np.ndarray, list[str]]:
    labels: list[int] = []
    paths: list[str] = []
    logits_batches: list[np.ndarray] = []
    for images, batch_labels, batch_paths in loader:
        logits = runner.infer(images)
        if logits.ndim != 2 or logits.shape[1] != len(classes):
            raise ValueError(f"Expected logits [batch,{len(classes)}], received {logits.shape}")
        logits_batches.append(logits)
        labels.extend(batch_labels.tolist())
        paths.extend(batch_paths)
    all_logits = np.concatenate(logits_batches, axis=0)
    all_labels = np.asarray(labels, dtype=np.int64)
    metrics = classification_metrics(all_labels, all_logits.argmax(axis=1), classes)
    return metrics, all_labels, all_logits, paths


def path_fingerprint(paths: list[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def macro_summary(domains: dict) -> dict:
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1_present_classes"]
    return {
        name: float(np.mean([payload["metrics"][name] for payload in domains.values()]))
        for name in metric_names
    }


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.img_size != DEFAULT_IMG_SIZE:
        raise ValueError(f"This student was trained for {DEFAULT_IMG_SIZE}x{DEFAULT_IMG_SIZE} input.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = get_dataset_specs(args.dataset_root)
    student_model, processor, metadata = load_checkpoint_model(args.student_pt, device)
    classes = metadata["classes"]
    transform = build_eval_transform(args.img_size, processor.image_mean, processor.image_std)

    results = {
        "protocol": {
            "split": args.split,
            "classes": classes,
            "img_size": args.img_size,
            "domains": list(specs),
            "domain_aggregation": "equal-weight macro average",
            "source_git_commit": git_commit(ROOT),
        },
        "models": {},
        "student_runtime_agreement": {},
    }
    student_outputs: dict[str, dict[str, tuple[np.ndarray, np.ndarray, list[str]]]] = {}

    def evaluate_runner(name: str, runner, domain_keys: list[str], family: str, artifact: Path) -> None:
        domain_results = {}
        captured = {}
        for domain in domain_keys:
            loader = build_loader(specs[domain], domain, args.split, classes, transform, args)
            metrics, labels, logits, paths = run_domain(runner, loader, classes)
            domain_results[domain] = {
                "metrics": metrics,
                "ordered_path_sha256": path_fingerprint(paths),
            }
            captured[domain] = (labels, logits, paths)
            print(
                f"{name}/{domain}: accuracy={metrics['accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1_present_classes']:.4f} n={metrics['samples']}"
            )
        results["models"][name] = {
            "family": family,
            "artifact": str(artifact),
            "artifact_sha256": sha256_file(artifact),
            "domains": domain_results,
            "domain_macro": macro_summary(domain_results),
        }
        if family == "student":
            student_outputs[name] = captured

    evaluate_runner(
        "student_pt",
        PyTorchRunner(student_model, device),
        list(specs),
        "student",
        args.student_pt,
    )
    del student_model
    release_cuda()

    if args.student_onnx:
        validate_manifest(args.student_onnx, classes, args.img_size, args.student_pt)
        onnx_runner = OnnxRuntimeRunner(args.student_onnx, use_cuda=device.type == "cuda")
        evaluate_runner("student_onnx", onnx_runner, list(specs), "student", args.student_onnx)
        results["models"]["student_onnx"]["providers"] = onnx_runner.providers
        del onnx_runner
        release_cuda()

    if args.student_engine:
        validate_manifest(args.student_engine, classes, args.img_size, args.student_pt)
        engine_runner = TensorRTRunner(args.student_engine, device)
        evaluate_runner("student_engine", engine_runner, list(specs), "student", args.student_engine)
        del engine_runner
        release_cuda()

    teacher_paths = {
        "cv": args.teacher_cv,
        "rs": args.teacher_rs,
        "uav": args.teacher_uav,
    }
    for domain, checkpoint in teacher_paths.items():
        if checkpoint is None:
            continue
        teacher, teacher_processor, teacher_metadata = load_checkpoint_model(checkpoint, device)
        if teacher_metadata["classes"] != classes:
            raise ValueError(f"Teacher {domain} class order differs from the student.")
        teacher_transform = build_eval_transform(
            args.img_size, teacher_processor.image_mean, teacher_processor.image_std
        )
        previous_transform = transform
        transform = teacher_transform
        evaluate_runner(
            f"teacher_{domain}",
            PyTorchRunner(teacher, device),
            [domain],
            "teacher",
            checkpoint,
        )
        transform = previous_transform
        del teacher
        release_cuda()

    reference = student_outputs["student_pt"]
    for candidate_name, candidate in student_outputs.items():
        if candidate_name == "student_pt":
            continue
        by_domain = {}
        for domain in specs:
            ref_labels, ref_logits, ref_paths = reference[domain]
            candidate_labels, candidate_logits, candidate_paths = candidate[domain]
            if ref_paths != candidate_paths or not np.array_equal(ref_labels, candidate_labels):
                raise RuntimeError(f"Ordered evaluation samples changed for {candidate_name}/{domain}.")
            by_domain[domain] = prediction_agreement(ref_logits, candidate_logits)
        results["student_runtime_agreement"][candidate_name] = by_domain

    write_json(args.output, results)
    print(f"Saved classification evaluation: {args.output}")


if __name__ == "__main__":
    main()
