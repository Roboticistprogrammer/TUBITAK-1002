"""
Evaluate models on detection task: compute mAP, Recall, F1 (patched).
Supports:
  - PyTorch (.pt) models
  - ONNX (.onnx) models
  - TensorRT (.engine) models
  
Outputs per-domain metrics and aggregate results as JSON.
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from ultralytics import YOLO


CLASS_NAMES = ["fire", "smoke", "neither"]
DOMAINS = ["FASDD_CV", "FASDD_RS", "FASDD_UAV"]


def resolve_domain_root(dataset_root: Path, domain: str) -> Optional[Path]:
    candidates = [
        dataset_root / domain,
        dataset_root / domain / domain,
    ]
    for candidate in candidates:
        if (candidate / "images").exists() and (candidate / "annotations").exists():
            return candidate
    return None


def build_domain_yaml(domain_root: Path, domain: str, temp_dir: Path) -> Path:
    if domain == "FASDD_CV":
        yolo_root = domain_root / "annotations" / "YOLO_CV"
    elif domain == "FASDD_RS":
        yolo_root = domain_root / "annotations" / "YOLO_RS_RGB"
    elif domain == "FASDD_UAV":
        yolo_root = domain_root / "annotations" / "YOLO_UAV"
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    labels_dir = yolo_root / "labels"
    images_dir = domain_root / "images"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Missing images/labels for {domain} at {domain_root}")

    train_txt = yolo_root / "train.txt"
    val_txt = yolo_root / "val.txt"
    test_txt = yolo_root / "test.txt"

    train_entry = str(train_txt) if train_txt.exists() else "images"
    val_entry = str(val_txt) if val_txt.exists() else train_entry
    test_entry = str(test_txt) if test_txt.exists() else val_entry

    yaml_lines = [
        f"path: {domain_root.resolve()}",
        f"train: {train_entry}",
        f"val: {val_entry}",
        f"test: {test_entry}",
        "names:",
    ]
    for index, name in enumerate(CLASS_NAMES):
        yaml_lines.append(f"  {index}: {name}")

    yaml_path = temp_dir / f"{domain.lower()}_data.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return yaml_path


def metric_value(metric_box, attribute: str) -> float:
    value = getattr(metric_box, attribute, 0.0)
    return float(value if value is not None else 0.0)


def compute_metrics_per_domain(model: YOLO, domain_root: Path, domain: str, imgsz: int, batch: int, device: str) -> Dict:
    with tempfile.TemporaryDirectory(prefix=f"eval_{domain.lower()}_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        data_yaml = build_domain_yaml(domain_root, domain, temp_dir)

        metrics = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=imgsz,
            batch=batch,
            conf=0.001,
            iou=0.6,
            device=device,
            verbose=False,
            plots=False,
            save_json=False,
            workers=0,
        )

    precision = metric_value(metrics.box, "mp")
    recall = metric_value(metrics.box, "mr")
    m_ap50 = metric_value(metrics.box, "map50")
    m_ap50_95 = metric_value(metrics.box, "map")
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "mAP50": m_ap50,
        "mAP50_95": m_ap50_95,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
    }


def evaluate_model_all_domains(
    model_path: Path,
    dataset_root: Path,
    imgsz: int,
    batch: int,
    device: str,
) -> Dict:
    """
    Evaluate model on all three domains: CV, RS, UAV.
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_path.name}")
    print(f"{'='*70}")
    
    model = YOLO(str(model_path))

    domain_metrics = {}
    aggregate_metrics = {
        "mAP50": [],
        "mAP50_95": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    
    for domain in DOMAINS:
        domain_root = resolve_domain_root(dataset_root, domain)
        if domain_root is None:
            print(f"⚠ {domain} not found under {dataset_root}")
            domain_metrics[domain] = {"error": f"Domain not found under {dataset_root}"}
            continue

        try:
            metrics = compute_metrics_per_domain(
                model=model,
                domain_root=domain_root,
                domain=domain,
                imgsz=imgsz,
                batch=batch,
                device=device,
            )
            domain_metrics[domain] = metrics
            for key in aggregate_metrics:
                aggregate_metrics[key].append(metrics[key])
        except Exception as error:
            domain_metrics[domain] = {"error": str(error)}
    
    # Compute macro averages
    result = {
        "model_name": model_path.stem,
        "model_path": str(model_path),
        "domains": domain_metrics,
        "metrics": {
            key: float(np.mean(values)) if values else 0.0
            for key, values in aggregate_metrics.items()
        },
        "evaluated_domains": sum(1 for _, value in domain_metrics.items() if "error" not in value),
    }
    
    print(f"\nAggregate metrics:")
    for key, value in result["metrics"].items():
        print(f"  {key}: {value:.4f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection models on real dataset.")
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to model file (.pt, .onnx, or .engine)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["pt", "onnx", "engine"],
        help="Model format type (kept for pipeline compatibility)"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets"),
        help="Path to datasets root folder"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size")
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size")
    parser.add_argument("--device", type=str, default="", help="Ultralytics device string; empty means auto")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/eval_results.json"),
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    if not args.model_path:
        print("Error: --model-path is required")
        parser.print_help()
        return
    
    if not args.model_path.exists():
        print(f"Error: Model not found at {args.model_path}")
        return
    
    # Evaluate
    result = evaluate_model_all_domains(
        model_path=args.model_path,
        dataset_root=args.dataset_root,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    
    # Save result
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
