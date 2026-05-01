from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from bootstrap import setup_path

ROOT = setup_path()

from firecls.config import DEFAULT_CLASSES, DEFAULT_IMG_SIZE, DEFAULT_MODEL_NAME, get_dataset_specs
from firecls.data.dataset import ImageClassificationCSVDataset
from firecls.models.swinv2 import build_swinv2_classifier
from firecls.utils import accuracy_from_logits, to_device


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate student vs teacher checkpoints on a test/val split and compare weights safely."
	)
	parser.add_argument("--student", type=Path, required=True)
	parser.add_argument("--cv-teacher", type=Path, required=True)
	parser.add_argument("--rs-teacher", type=Path, required=True)
	parser.add_argument("--uav-teacher", type=Path, required=True)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
	parser.add_argument("--split", choices=["val", "test"], default="test")
	parser.add_argument("--mismatch-preview", type=int, default=5)
	return parser.parse_args()


def build_eval_transform(img_size: int, mean, std):
	return transforms.Compose(
		[
			transforms.Resize(img_size + 32),
			transforms.CenterCrop(img_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		]
	)


def load_checkpoint_model(checkpoint_path: Path):
	checkpoint = torch.load(checkpoint_path, map_location="cpu")
	classes = checkpoint.get("classes", DEFAULT_CLASSES)
	model_name = checkpoint.get("model_name", DEFAULT_MODEL_NAME)
	model, image_processor = build_swinv2_classifier(
		num_labels=len(classes),
		label2id={name: i for i, name in enumerate(classes)},
		id2label={i: name for i, name in enumerate(classes)},
		model_name=model_name,
	)
	model.load_state_dict(checkpoint["model"])
	model.eval()
	return model, image_processor, classes, model_name


def evaluate_accuracy(model, loader: DataLoader, device: torch.device) -> float:
	model.eval()
	total_acc = 0.0
	total = 0
	with torch.no_grad():
		for images, labels in loader:
			images, labels = to_device((images, labels), device)
			outputs = model(images)
			logits = getattr(outputs, "logits", outputs)
			total_acc += accuracy_from_logits(logits, labels) * labels.size(0)
			total += labels.size(0)
	return total_acc / max(1, total)


@dataclass
class WeightCompareStats:
	matched_tensors: int
	matched_numel: int
	missing_in_teacher: int
	missing_in_student: int
	mismatched_shapes: int
	cosine_similarity: float
	relative_l2: float
	sample_mismatches: list[str]


def compare_state_dicts(
	student_state: dict[str, torch.Tensor],
	teacher_state: dict[str, torch.Tensor],
	mismatch_preview: int = 5,
) -> WeightCompareStats:
	matched_tensors = 0
	matched_numel = 0
	mismatched_shapes = 0
	sample_mismatches: list[str] = []

	dot_sum = 0.0
	student_norm = 0.0
	teacher_norm = 0.0
	diff_sq_sum = 0.0

	for key, student_tensor in student_state.items():
		if key not in teacher_state:
			if len(sample_mismatches) < mismatch_preview:
				sample_mismatches.append(f"missing in teacher: {key}")
			continue
		teacher_tensor = teacher_state[key]
		if student_tensor.shape != teacher_tensor.shape:
			mismatched_shapes += 1
			if len(sample_mismatches) < mismatch_preview:
				sample_mismatches.append(
					f"shape mismatch: {key} {tuple(student_tensor.shape)} vs {tuple(teacher_tensor.shape)}"
				)
			continue

		s = student_tensor.float().flatten()
		t = teacher_tensor.float().flatten()
		dot_sum += torch.dot(s, t).item()
		student_norm += torch.dot(s, s).item()
		teacher_norm += torch.dot(t, t).item()
		diff = s - t
		diff_sq_sum += torch.dot(diff, diff).item()

		matched_tensors += 1
		matched_numel += s.numel()

	missing_in_student = 0
	for key in teacher_state.keys():
		if key not in student_state:
			missing_in_student += 1
			if len(sample_mismatches) < mismatch_preview:
				sample_mismatches.append(f"missing in student: {key}")

	denom = (student_norm**0.5) * (teacher_norm**0.5)
	cosine_similarity = dot_sum / denom if denom > 0 else 0.0
	relative_l2 = diff_sq_sum / student_norm if student_norm > 0 else 0.0

	missing_in_teacher = sum(1 for key in student_state.keys() if key not in teacher_state)

	return WeightCompareStats(
		matched_tensors=matched_tensors,
		matched_numel=matched_numel,
		missing_in_teacher=missing_in_teacher,
		missing_in_student=missing_in_student,
		mismatched_shapes=mismatched_shapes,
		cosine_similarity=cosine_similarity,
		relative_l2=relative_l2,
		sample_mismatches=sample_mismatches,
	)


def build_loader(
	index_csv: Path,
	split: str,
	classes: list[str],
	transform,
	batch_size: int,
	num_workers: int,
) -> DataLoader:
	dataset = ImageClassificationCSVDataset(index_csv, split, classes, transform=transform)
	return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def ensure_index_csv(domain_name: str) -> Path:
	index_csv = Path("data_index") / f"{domain_name.lower()}_tdml.csv"
	if not index_csv.exists():
		raise SystemExit(f"CSV not found: {index_csv}. Run prepare_tdml_classification.py first.")
	return index_csv


def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	student_model, student_processor, student_classes, student_model_name = load_checkpoint_model(args.student)
	student_model.to(device)

	teachers = {
		"cv": load_checkpoint_model(args.cv_teacher),
		"rs": load_checkpoint_model(args.rs_teacher),
		"uav": load_checkpoint_model(args.uav_teacher),
	}
	for teacher_model, _, _, _ in teachers.values():
		teacher_model.to(device)

	print("Weight comparison (student vs teacher):")
	student_state = student_model.state_dict()
	for domain, (teacher_model, _, _, _) in teachers.items():
		stats = compare_state_dicts(student_state, teacher_model.state_dict(), args.mismatch_preview)
		print(
			f"- {domain}: matched_tensors={stats.matched_tensors} matched_numel={stats.matched_numel} "
			f"missing_in_teacher={stats.missing_in_teacher} missing_in_student={stats.missing_in_student} "
			f"mismatched_shapes={stats.mismatched_shapes} cosine={stats.cosine_similarity:.4f} "
			f"rel_l2={stats.relative_l2:.6f}"
		)
		if stats.sample_mismatches:
			preview = " | ".join(stats.sample_mismatches)
			print(f"  mismatches: {preview}")

	specs = get_dataset_specs(ROOT)

	print("\nAccuracy comparison:")
	student_tf = build_eval_transform(
		args.img_size,
		student_processor.image_mean,
		student_processor.image_std,
	)
	for domain, spec in specs.items():
		index_csv = ensure_index_csv(spec.name)
		student_loader = build_loader(
			index_csv,
			args.split,
			student_classes,
			student_tf,
			args.batch_size,
			args.num_workers,
		)
		student_acc = evaluate_accuracy(student_model, student_loader, device)

		teacher_model, teacher_processor, teacher_classes, teacher_name = teachers[domain]
		teacher_tf = build_eval_transform(
			args.img_size,
			teacher_processor.image_mean,
			teacher_processor.image_std,
		)
		teacher_loader = build_loader(
			index_csv,
			args.split,
			teacher_classes,
			teacher_tf,
			args.batch_size,
			args.num_workers,
		)
		teacher_acc = evaluate_accuracy(teacher_model, teacher_loader, device)

		print(
			f"- {domain}: student_acc={student_acc:.4f} ({student_model_name}) "
			f"teacher_acc={teacher_acc:.4f} ({teacher_name})"
		)


if __name__ == "__main__":
	main()