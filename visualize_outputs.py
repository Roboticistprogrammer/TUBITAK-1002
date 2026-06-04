import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark_models import list_images, preprocess_image, build_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visual report outputs for ONNX/TensorRT models")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to .onnx or .engine model")
    parser.add_argument("--model-type", choices=["onnx", "engine"], required=True, help="Backend type")
    parser.add_argument("--sample-dir", type=Path, default=Path("sample"), help="Sample images root")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visuals"), help="Visualization output root")
    parser.add_argument("--input-size", type=int, default=224, help="Input square size")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--class-names", type=str, default="fire,neither,smoke", help="Comma-separated class labels")
    parser.add_argument("--max-images-per-class", type=int, default=30, help="Limit images per class per domain")
    parser.add_argument("--save-contact-sheet", action="store_true", help="Create a report-ready contact sheet")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU provider for ONNX")
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def class_name_from_idx(class_names: List[str], idx: int) -> str:
    if idx < len(class_names):
        return class_names[idx]
    return f"class_{idx}"


def annotate_image(image_bgr: np.ndarray, lines: List[str]) -> np.ndarray:
    annotated = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    pad = 8
    line_height = 22
    box_height = pad * 2 + line_height * len(lines)

    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], box_height), (0, 0, 0), -1)
    y = pad + 16
    for line in lines:
        cv2.putText(annotated, line, (10, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        y += line_height

    return annotated


def select_images(sample_root: Path, max_per_class: int) -> List[Path]:
    selected = []
    for domain in sorted([p for p in sample_root.iterdir() if p.is_dir()]):
        for class_dir in sorted([p for p in domain.iterdir() if p.is_dir()]):
            class_images = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])
            selected.extend(class_images[:max_per_class])
    if not selected:
        selected = list_images(sample_root)
    return selected


def create_contact_sheet(items: List[dict], output_path: Path, thumb_size=(360, 220), cols: int = 3) -> None:
    if not items:
        return

    rows = (len(items) + cols - 1) // cols
    caption_h = 50
    canvas_w = cols * thumb_size[0]
    canvas_h = rows * (thumb_size[1] + caption_h)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, item in enumerate(items):
        row, col = divmod(idx, cols)
        x = col * thumb_size[0]
        y = row * (thumb_size[1] + caption_h)

        image = Image.open(item["output_path"]).convert("RGB")
        image = image.resize(thumb_size)
        canvas.paste(image, (x, y))

        caption = f"{item['domain']}/{item['gt_class']} -> {item['pred_class']} ({item['confidence']*100:.1f}%)"
        draw.text((x + 6, y + thumb_size[1] + 8), caption, fill=(230, 230, 230), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {args.sample_dir}")

    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    backend = build_backend(args)

    images = select_images(args.sample_dir, args.max_images_per_class)
    if not images:
        raise ValueError(f"No images found in {args.sample_dir}")

    model_name = args.model_path.stem
    run_dir = args.output_dir / model_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for image_path in images:
        image_chw = preprocess_image(image_path, args.input_size)
        batch = np.expand_dims(image_chw, axis=0).astype(np.float32)

        start = time.perf_counter()
        logits = backend.infer(batch)[0]
        latency_ms = (time.perf_counter() - start) * 1000.0

        probabilities = softmax(logits.astype(np.float32))
        pred_idx = int(np.argmax(probabilities))
        pred_class = class_name_from_idx(class_names, pred_idx)
        confidence = float(probabilities[pred_idx])

        rel = image_path.relative_to(args.sample_dir)
        parts = rel.parts
        domain = parts[0] if len(parts) > 0 else "unknown"
        gt_class = parts[1] if len(parts) > 1 else "unknown"

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        lines = [
            f"GT: {gt_class}",
            f"Pred: {pred_class} ({confidence * 100:.2f}%)",
            f"Latency: {latency_ms:.2f} ms",
        ]
        annotated = annotate_image(image_bgr, lines)

        output_path = run_dir / rel
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)

        records.append(
            {
                "image": str(rel),
                "domain": domain,
                "gt_class": gt_class,
                "pred_idx": pred_idx,
                "pred_class": pred_class,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "output_path": str(output_path),
            }
        )

    with open(run_dir / "predictions.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model_path": str(args.model_path),
                "model_type": args.model_type,
                "class_names": class_names,
                "num_images": len(records),
                "predictions": records,
            },
            file,
            indent=2,
        )

    with open(run_dir / "predictions.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "domain", "gt_class", "pred_idx", "pred_class", "confidence", "latency_ms", "output_path"],
        )
        writer.writeheader()
        writer.writerows(records)

    if args.save_contact_sheet:
        create_contact_sheet(records[: min(24, len(records))], run_dir / "contact_sheet.png")

    print("=" * 72)
    print(f"Saved visual outputs: {run_dir}")
    print(f"Total images processed: {len(records)}")
    print(f"Mean latency: {np.mean([r['latency_ms'] for r in records]):.3f} ms")
    print("=" * 72)


if __name__ == "__main__":
    main()