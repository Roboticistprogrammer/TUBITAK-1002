"""
Calculate mAP, F1, Recall, and Accuracy metrics for teacher and student .pt models.

This script:
1. Loads all .pt model files (teachers and student)
2. Evaluates them on YOLO format validation datasets
3. Extracts mAP, F1, Recall, and Accuracy metrics
4. Outputs results to benchmark_metrics.json for use in figure_generator

Evaluation strategy:
- Teachers: Evaluated on their own domain's validation dataset
- Student: Cross-domain evaluation on all three domains (CV, RS, UAV)

Usage:
    python Benchmark_models.py --output results/benchmark_metrics.json
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import subprocess

import torch
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelBenchmarker:
    """Benchmark .pt models for object detection metrics using YOLO format datasets."""
    
    def __init__(self, models_dir: Path = Path("Models"), datasets_dir: Path = Path("datasets")):
        self.models_dir = models_dir
        self.datasets_dir = datasets_dir
        self.results = {
            "experiment": {
                "title": "Teacher-Student Model Benchmarking",
                "hardware": "Host Machine",
                "timestamp": datetime.now().isoformat(),
            },
            "models": [],
            "summary": {}
        }
    
    def load_model(self, model_path: Path) -> Optional[torch.nn.Module]:
        """Load a YOLOv5 .pt model."""
        try:
            logger.info(f"Loading model from {model_path}")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path), force_reload=False)
            model.conf = 0.25  # Set confidence threshold
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return None
    
    def create_yolo_dataset_yaml(self, domain: str, yolo_dir: Path) -> Optional[Path]:
        """Create a temporary YOLO dataset.yaml file for evaluation."""
        try:
            yaml_path = Path(f"/tmp/dataset_{domain}.yaml")
            
            dataset_config = {
                "path": str(yolo_dir.parent),
                "train": str(yolo_dir / "train.txt"),
                "val": str(yolo_dir / "val.txt"),
                "test": str(yolo_dir / "test.txt"),
                "nc": 2,  # number of classes
                "names": ["Fire", "Smoke"]  # class names
            }
            
            with open(yaml_path, 'w') as f:
                yaml.dump(dataset_config, f)
            
            return yaml_path
        except Exception as e:
            logger.error(f"Failed to create dataset YAML for {domain}: {e}")
            return None
    
    def evaluate_model_on_dataset(self, model_path: Path, domain: str, yolo_dir: Path) -> Dict[str, float]:
        """Evaluate a model on a YOLO format dataset and extract metrics."""
        metrics = {}
        
        try:
            # Check if dataset exists
            if not yolo_dir.exists():
                logger.warning(f"YOLO dataset not found for {domain} at {yolo_dir}")
                return metrics
            
            # Load model
            model = self.load_model(model_path)
            if model is None:
                return metrics
            
            # Create dataset YAML
            yaml_path = self.create_yolo_dataset_yaml(domain, yolo_dir)
            if yaml_path is None:
                return metrics
            
            logger.info(f"Evaluating {model_path.parent.name} on {domain} dataset")
            
            # Run validation using YOLOv5's validate function
            results = model.val(
                data=str(yaml_path),
                imgsz=640,
                batch=16,
                conf=0.25,
                iou=0.6,
                device=0 if torch.cuda.is_available() else 'cpu',
                verbose=False,
            )
            
            # Extract metrics from results
            if results is not None:
                metrics = {
                    "mAP50": float(results.results_dict.get("metrics/mAP50", 0.0)),
                    "mAP50_95": float(results.results_dict.get("metrics/mAP50-95", 0.0)),
                    "precision": float(results.results_dict.get("metrics/precision", 0.0)),
                    "recall": float(results.results_dict.get("metrics/recall", 0.0)),
                    "f1": float(results.results_dict.get("metrics/f1", 0.0)),
                }
                logger.info(f"Metrics for {domain}: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model on {domain}: {e}")
            return metrics
    
    def extract_history_metrics(self, history_path: Path) -> Dict[str, float]:
        """Extract training metrics from history JSON."""
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            records = history.get("records", [])
            if not records:
                logger.warning(f"No records found in {history_path}")
                return {}
            
            final_record = records[-1]
            
            # For teacher models (single domain)
            if "val_acc" in final_record and isinstance(final_record["val_acc"], (int, float)):
                return {
                    "train_accuracy": float(final_record.get("train_acc", 0.0)),
                    "val_accuracy": float(final_record.get("val_acc", 0.0)),
                    "train_loss": float(final_record.get("train_loss", 0.0)),
                }
            
            # For student models (multi-domain)
            elif "val_acc" in final_record and isinstance(final_record["val_acc"], dict):
                val_acc = final_record["val_acc"]
                return {
                    "val_accuracy_cv": float(val_acc.get("cv", 0.0)),
                    "val_accuracy_rs": float(val_acc.get("rs", 0.0)),
                    "val_accuracy_uav": float(val_acc.get("uav", 0.0)),
                    "val_accuracy_avg": float(final_record.get("avg_val", 0.0)),
                    "train_accuracy": float(final_record.get("train_acc", 0.0)),
                    "train_loss": float(final_record.get("train_loss", 0.0)),
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error extracting history metrics from {history_path}: {e}")
            return {}
    
    def benchmark_teachers(self) -> List[Dict]:
        """Benchmark all teacher models on their own domains."""
        teacher_models = []
        teachers_dir = self.models_dir / "teachers"
        
        if not teachers_dir.exists():
            logger.warning(f"Teachers directory not found: {teachers_dir}")
            return teacher_models
        
        for domain_dir in sorted(teachers_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            
            domain = domain_dir.name
            best_pt = domain_dir / "best.pt"
            last_pt = domain_dir / "last.pt"
            history_json = domain_dir / "history.json"
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing teacher model: {domain}")
            logger.info(f"{'='*60}")
            
            model_data = {
                "name": f"Teacher_{domain}",
                "type": "teacher",
                "domain": domain,
                "paths": {
                    "best": str(best_pt) if best_pt.exists() else None,
                    "last": str(last_pt) if last_pt.exists() else None,
                    "history": str(history_json) if history_json.exists() else None,
                }
            }
            
            # Extract training metrics from history
            if history_json.exists():
                history_metrics = self.extract_history_metrics(history_json)
                if history_metrics:
                    model_data["history_metrics"] = history_metrics
            
            # Evaluate on YOLO dataset (use best.pt if available)
            eval_model = best_pt if best_pt.exists() else last_pt
            if eval_model.exists():
                yolo_dir = self.datasets_dir / domain / "YOLO" / f"{domain[6:]}_YOLO"
                if yolo_dir.exists():
                    eval_metrics = self.evaluate_model_on_dataset(eval_model, domain, yolo_dir)
                    if eval_metrics:
                        model_data["evaluation_metrics"] = eval_metrics
                else:
                    logger.warning(f"YOLO dataset not found at {yolo_dir}")
            
            teacher_models.append(model_data)
        
        return teacher_models
    
    def benchmark_student(self) -> Optional[Dict]:
        """Benchmark student model on all three domains (cross-domain evaluation)."""
        student_dir = self.models_dir / "students"
        
        if not student_dir.exists():
            logger.warning(f"Students directory not found: {student_dir}")
            return None
        
        last_pt = student_dir / "last.pt"
        history_json = student_dir / "history.json"
        
        logger.info(f"\n{'='*60}")
        logger.info("Processing student model (cross-domain evaluation)")
        logger.info(f"{'='*60}")
        
        model_data = {
            "name": "Student",
            "type": "student",
            "paths": {
                "last": str(last_pt) if last_pt.exists() else None,
                "history": str(history_json) if history_json.exists() else None,
            },
            "cross_domain_evaluations": {}
        }
        
        # Extract training metrics from history
        if history_json.exists():
            history_metrics = self.extract_history_metrics(history_json)
            if history_metrics:
                model_data["history_metrics"] = history_metrics
        
        # Cross-domain evaluation: evaluate student on all three domains
        if last_pt.exists():
            domains = ["FASDD_CV", "FASDD_RS", "FASDD_UAV"]
            
            for domain in domains:
                yolo_dir = self.datasets_dir / domain / "YOLO" / f"{domain[6:]}_YOLO"
                
                if yolo_dir.exists():
                    logger.info(f"\nEvaluating student on {domain} domain...")
                    eval_metrics = self.evaluate_model_on_dataset(last_pt, domain, yolo_dir)
                    
                    if eval_metrics:
                        model_data["cross_domain_evaluations"][domain] = eval_metrics
                else:
                    logger.warning(f"YOLO dataset not found at {yolo_dir}")
        
        return model_data
    
    def run_benchmark(self) -> Dict:
        """Run complete benchmark suite."""
        logger.info("Starting model benchmarking...")
        
        # Benchmark teachers
        teacher_models = self.benchmark_teachers()
        self.results["models"].extend(teacher_models)
        
        # Benchmark student
        student_model = self.benchmark_student()
        if student_model:
            self.results["models"].append(student_model)
        
        # Generate summary
        self._generate_summary()
        
        logger.info(f"Benchmarking complete. Found {len(self.results['models'])} models.")
        return self.results
    
    def _generate_summary(self) -> None:
        """Generate summary statistics across models."""
        summary = {}
        
        for model in self.results["models"]:
            model_type = model.get("type", "unknown")
            model_name = model.get("name", "unknown")
            
            if model_type == "teacher":
                domain = model.get("domain", "unknown")
                
                # Add history metrics
                if "history_metrics" in model:
                    for key, value in model["history_metrics"].items():
                        summary[f"{domain}_{key}"] = value
                
                # Add evaluation metrics
                if "evaluation_metrics" in model:
                    for key, value in model["evaluation_metrics"].items():
                        summary[f"{domain}_{key}"] = value
            
            elif model_type == "student":
                # Add history metrics
                if "history_metrics" in model:
                    for key, value in model["history_metrics"].items():
                        summary[f"student_{key}"] = value
                
                # Add cross-domain evaluation summary
                if "cross_domain_evaluations" in model:
                    for domain, metrics in model["cross_domain_evaluations"].items():
                        for key, value in metrics.items():
                            summary[f"student_{domain}_{key}"] = value
        
        self.results["summary"] = summary
    
    def save_results(self, output_path: Path) -> None:
        """Save benchmark results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark .pt models with YOLO evaluation")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark_metrics.json"),
        help="Output JSON file path"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("Models"),
        help="Directory containing models"
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("datasets"),
        help="Directory containing datasets"
    )
    
    args = parser.parse_args()
    
    benchmarker = ModelBenchmarker(
        models_dir=args.models_dir,
        datasets_dir=args.datasets_dir
    )
    
    results = benchmarker.run_benchmark()
    benchmarker.save_results(args.output)
    
    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    for model in results["models"]:
        print(f"\n{model['name']}:")
        
        # Print history metrics
        if "history_metrics" in model:
            print("  History Metrics:")
            for key, value in model["history_metrics"].items():
                if value is not None:
                    print(f"    {key}: {value:.4f}")
        
        # Print evaluation metrics for teachers
        if "evaluation_metrics" in model:
            print("  Evaluation Metrics:")
            for key, value in model["evaluation_metrics"].items():
                if value is not None:
                    print(f"    {key}: {value:.4f}")
        
        # Print cross-domain evaluation for student
        if "cross_domain_evaluations" in model:
            print("  Cross-Domain Evaluations:")
            for domain, metrics in model["cross_domain_evaluations"].items():
                print(f"    {domain}:")
                for key, value in metrics.items():
                    if value is not None:
                        print(f"      {key}: {value:.4f}")
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    for key, value in results["summary"].items():
        if value is not None:
            print(f"{key}: {value:.4f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()