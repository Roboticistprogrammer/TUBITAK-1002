import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Export an Ultralytics YOLO model and place the artifact in the Triton repository."
	)
	parser.add_argument(
		"--model",
		default="model.pt",
		help="Path to the source model file (default: model.pt).",
	)
	parser.add_argument(
		"--format",
		default="onnx",
		choices=["onnx", "engine"],
		help="Export format (default: onnx).",
	)
	parser.add_argument(
		"--output-dir",
		default=str(Path(__file__).resolve().parents[1] / "model_repository" / "swin-tf" / "1"),
		help="Target directory for exported model files.",
	)
	parser.add_argument(
		"--imgsz",
		type=int,
		default=640,
		help="Input image size for export (default: 640).",
	)
	parser.add_argument(
		"--dynamic",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Enable dynamic shapes where supported.",
	)
	parser.add_argument(
		"--half",
		action="store_true",
		help="Export using FP16 where supported.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()

	model_path = Path(args.model)
	if not model_path.exists():
		raise FileNotFoundError(f"Model file not found: {model_path}")

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	model = YOLO(str(model_path))
	exported_path = model.export(
		format=args.format,
		imgsz=args.imgsz,
		dynamic=args.dynamic,
		half=args.half,
	)

	exported_path = Path(exported_path)
	target_path = output_dir / exported_path.name
	if exported_path.resolve() != target_path.resolve():
		shutil.move(str(exported_path), str(target_path))

	print(f"Exported model saved to: {target_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())