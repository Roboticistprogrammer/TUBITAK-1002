"""
Generate report-ready visualizations from model training history files.

Inputs used (only):
  - Models/teachers/FASDD_CV/history.json
  - Models/teachers/FASDD_RS/history.json
  - Models/teachers/FASDD_UAV/history.json
  - Models/students/history.json
  - Models/comparison.txt

Outputs:
  - figure_training_overview.png
  - figure_domain_comparison.png
  - figure_comparison_table.png
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_history(path: Path) -> List[Dict]:
	with path.open("r", encoding="utf-8") as file:
		payload = json.load(file)
	return payload.get("records", [])


def parse_comparison(path: Path) -> Dict[str, Dict[str, float]]:
	text = path.read_text(encoding="utf-8")
	parsed = {}
	pattern = re.compile(r"-\s*(\w+):\s*student_acc=([0-9.]+).*?teacher_acc=([0-9.]+)", re.IGNORECASE)
	for domain, student_acc, teacher_acc in pattern.findall(text):
		parsed[domain.lower()] = {
			"student": float(student_acc),
			"teacher": float(teacher_acc),
		}
	return parsed


def teacher_series(records: List[Dict]) -> Tuple[List[int], List[float], List[float], List[float]]:
	epochs = [int(item.get("epoch", index)) for index, item in enumerate(records)]
	train_loss = [float(item.get("train_loss", 0.0)) for item in records]
	train_acc = [float(item.get("train_acc", 0.0)) for item in records]
	val_acc = [float(item.get("val_acc", 0.0)) for item in records]
	return epochs, train_loss, train_acc, val_acc


def student_series(records: List[Dict]) -> Dict[str, List[float]]:
	epochs = [int(item.get("epoch", index + 1)) for index, item in enumerate(records)]
	train_loss = [float(item.get("train_loss", 0.0)) for item in records]
	train_acc = [float(item.get("train_acc", 0.0)) for item in records]
	avg_val = [float(item.get("avg_val", 0.0)) for item in records]

	val_cv = [float(item.get("val_acc", {}).get("cv", 0.0)) for item in records]
	val_rs = [float(item.get("val_acc", {}).get("rs", 0.0)) for item in records]
	val_uav = [float(item.get("val_acc", {}).get("uav", 0.0)) for item in records]

	return {
		"epochs": epochs,
		"train_loss": train_loss,
		"train_acc": train_acc,
		"avg_val": avg_val,
		"val_cv": val_cv,
		"val_rs": val_rs,
		"val_uav": val_uav,
	}


def plot_training_overview(
	teacher_data: Dict[str, List[Dict]],
	student_records: List[Dict],
	output_path: Path,
) -> None:
	plt.style.use("seaborn-v0_8-whitegrid")
	fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=200)
	fig.suptitle("Training Overview: Teacher Models and Student Distillation", fontsize=16, fontweight="bold")

	domain_colors = {
		"FASDD_CV": "#1f77b4",
		"FASDD_RS": "#2ca02c",
		"FASDD_UAV": "#ff7f0e",
	}

	ax_loss = axes[0, 0]
	for domain, records in teacher_data.items():
		epochs, train_loss, _, _ = teacher_series(records)
		ax_loss.plot(epochs, train_loss, label=f"Teacher {domain}", color=domain_colors[domain], linewidth=2)
	student = student_series(student_records)
	ax_loss.plot(student["epochs"], student["train_loss"], label="Student", color="#d62728", linewidth=2.2, linestyle="--")
	ax_loss.set_title("Training Loss vs Epoch")
	ax_loss.set_xlabel("Epoch")
	ax_loss.set_ylabel("Loss")
	ax_loss.legend(fontsize=9)

	ax_acc = axes[0, 1]
	for domain, records in teacher_data.items():
		epochs, _, train_acc, _ = teacher_series(records)
		ax_acc.plot(epochs, train_acc, label=f"Teacher {domain}", color=domain_colors[domain], linewidth=2)
	ax_acc.plot(student["epochs"], student["train_acc"], label="Student", color="#d62728", linewidth=2.2, linestyle="--")
	ax_acc.set_title("Training Accuracy vs Epoch")
	ax_acc.set_xlabel("Epoch")
	ax_acc.set_ylabel("Accuracy")
	ax_acc.legend(fontsize=9)

	ax_teacher_val = axes[1, 0]
	for domain, records in teacher_data.items():
		epochs, _, _, val_acc = teacher_series(records)
		ax_teacher_val.plot(epochs, val_acc, label=f"Teacher {domain}", color=domain_colors[domain], linewidth=2)
	ax_teacher_val.set_title("Teacher Validation Accuracy")
	ax_teacher_val.set_xlabel("Epoch")
	ax_teacher_val.set_ylabel("Validation Accuracy")
	ax_teacher_val.legend(fontsize=9)

	ax_student_val = axes[1, 1]
	ax_student_val.plot(student["epochs"], student["avg_val"], label="Student Avg Val", color="#d62728", linewidth=2.5)
	ax_student_val.plot(student["epochs"], student["val_cv"], label="Student Val CV", color="#1f77b4", linewidth=1.8, linestyle=":")
	ax_student_val.plot(student["epochs"], student["val_rs"], label="Student Val RS", color="#2ca02c", linewidth=1.8, linestyle=":")
	ax_student_val.plot(student["epochs"], student["val_uav"], label="Student Val UAV", color="#ff7f0e", linewidth=1.8, linestyle=":")
	ax_student_val.set_title("Student Validation Accuracy")
	ax_student_val.set_xlabel("Epoch")
	ax_student_val.set_ylabel("Validation Accuracy")
	ax_student_val.legend(fontsize=9)

	fig.tight_layout(rect=[0, 0, 1, 0.96])
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)


def plot_domain_comparison(comparison: Dict[str, Dict[str, float]], output_path: Path) -> None:
	plt.style.use("seaborn-v0_8-whitegrid")
	domains = ["cv", "rs", "uav"]
	labels = [domain.upper() for domain in domains]
	teacher = [comparison[domain]["teacher"] for domain in domains]
	student = [comparison[domain]["student"] for domain in domains]
	gap = [teacher[index] - student[index] for index in range(len(domains))]

	fig, ax = plt.subplots(figsize=(10.5, 6), dpi=220)
	positions = list(range(len(domains)))
	width = 0.34

	teacher_bars = ax.bar([x - width / 2 for x in positions], teacher, width=width, label="Teacher", color="#1f77b4")
	student_bars = ax.bar([x + width / 2 for x in positions], student, width=width, label="Student", color="#d62728")

	for bar in list(teacher_bars) + list(student_bars):
		height = bar.get_height()
		ax.text(bar.get_x() + bar.get_width() / 2, height + 0.002, f"{height:.4f}", ha="center", va="bottom", fontsize=9)

	for idx, value in enumerate(gap):
		y_pos = max(teacher[idx], student[idx]) + 0.015
		ax.text(idx, y_pos, f"Gap: {value:.4f}", ha="center", va="bottom", fontsize=9, color="#333333")

	ax.set_xticks(positions)
	ax.set_xticklabels(labels)
	ax.set_ylim(0.70, 1.00)
	ax.set_ylabel("Accuracy")
	ax.set_title("Teacher vs Student Accuracy by Domain")
	ax.legend()

	fig.tight_layout()
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)


def plot_comparison_table(comparison: Dict[str, Dict[str, float]], output_path: Path) -> None:
	domains = ["cv", "rs", "uav"]
	rows = []
	for domain in domains:
		teacher_acc = comparison[domain]["teacher"]
		student_acc = comparison[domain]["student"]
		rows.append([
			domain.upper(),
			f"{teacher_acc:.4f}",
			f"{student_acc:.4f}",
			f"{teacher_acc - student_acc:.4f}",
		])

	teacher_avg = sum(comparison[d]["teacher"] for d in domains) / len(domains)
	student_avg = sum(comparison[d]["student"] for d in domains) / len(domains)
	rows.append(["AVG", f"{teacher_avg:.4f}", f"{student_avg:.4f}", f"{(teacher_avg - student_avg):.4f}"])

	fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=240)
	ax.axis("off")
	table = ax.table(
		cellText=rows,
		colLabels=["Domain", "Teacher Acc", "Student Acc", "Gap (T-S)"],
		loc="center",
		cellLoc="center",
		colLoc="center",
	)
	table.auto_set_font_size(False)
	table.set_fontsize(10)
	table.scale(1.1, 1.5)

	for (row, col), cell in table.get_celld().items():
		if row == 0:
			cell.set_text_props(weight="bold", color="white")
			cell.set_facecolor("#2f4f6f")
		elif row == len(rows):
			cell.set_text_props(weight="bold")
			cell.set_facecolor("#e8f0f8")
		else:
			cell.set_facecolor("#f8f9fb")

	ax.set_title("Domain Accuracy Summary Table", fontsize=13, fontweight="bold", pad=14)
	fig.tight_layout()
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)


def build_paths(base_dir: Path) -> Dict[str, Path]:
	return {
		"teacher_cv": base_dir / "teachers" / "FASDD_CV" / "history.json",
		"teacher_rs": base_dir / "teachers" / "FASDD_RS" / "history.json",
		"teacher_uav": base_dir / "teachers" / "FASDD_UAV" / "history.json",
		"student": base_dir / "students" / "history.json",
		"comparison": base_dir / "comparison.txt",
	}


def validate_inputs(paths: Dict[str, Path]) -> None:
	missing = [str(path) for path in paths.values() if not path.exists()]
	if missing:
		raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate professional report figures from training history.")
	parser.add_argument("--base-dir", type=Path, default=Path("Models"), help="Base folder containing teachers/, students/, and comparison.txt")
	parser.add_argument("--output-dir", type=Path, default=Path("results/figures"), help="Directory to save generated figures")
	args = parser.parse_args()

	paths = build_paths(args.base_dir)
	validate_inputs(paths)

	args.output_dir.mkdir(parents=True, exist_ok=True)

	teacher_data = {
		"FASDD_CV": load_history(paths["teacher_cv"]),
		"FASDD_RS": load_history(paths["teacher_rs"]),
		"FASDD_UAV": load_history(paths["teacher_uav"]),
	}
	student_data = load_history(paths["student"])
	comparison = parse_comparison(paths["comparison"])

	required_domains = {"cv", "rs", "uav"}
	if set(comparison.keys()) != required_domains:
		raise ValueError(f"comparison.txt should contain domains {sorted(required_domains)}")

	training_fig = args.output_dir / "figure_training_overview.png"
	comparison_fig = args.output_dir / "figure_domain_comparison.png"
	table_fig = args.output_dir / "figure_comparison_table.png"

	plot_training_overview(teacher_data, student_data, training_fig)
	plot_domain_comparison(comparison, comparison_fig)
	plot_comparison_table(comparison, table_fig)

	print("Generated figures:")
	print(f"  - {training_fig}")
	print(f"  - {comparison_fig}")
	print(f"  - {table_fig}")


if __name__ == "__main__":
	main()
