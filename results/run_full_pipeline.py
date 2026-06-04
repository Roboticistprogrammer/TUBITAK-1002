"""
Orchestrate full evaluation pipeline:
  1. Run evaluate_models.py on each model (.pt, .onnx, .engine)
  2. Run benchmark_models.py on each model
  3. Merge results into benchmark_metrics.json for plotting
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
import re


def safe_model_id(model_name: str) -> str:
    """Sanitize model name for use in filenames."""
    # Replace spaces and special chars with underscores
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
    return safe_name.lower()


def detect_python_executable() -> str:
    candidates = [
        Path.cwd() / ".venv" / "bin" / "python",
        Path.cwd() / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def run_evaluation(model_path: Path, model_type: str, dataset_root: Path, python_executable: str, model_id: str = None) -> Dict:
    """Run evaluation script and return results."""
    if model_id is None:
        model_id = model_path.stem
    else:
        model_id = safe_model_id(model_id)
    
    print(f"\n--- Running evaluation for {model_path.name} ---")
    
    result_file = Path("results") / f"eval_{model_id}.json"
    
    cmd = [
        python_executable,
        "results/evaluate_models.py",
        "--model-path", str(model_path),
        "--model-type", model_type,
        "--dataset-root", str(dataset_root),
        "--output-json", str(result_file),
    ]
    
    try:
        subprocess.run(cmd, check=True)
        with open(result_file, 'r') as f:
            return json.load(f)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Evaluation failed: {e}")
        return None


def run_benchmark(model_path: Path, model_type: str, python_executable: str, model_id: str = None) -> Dict:
    """Run benchmark script and return results."""
    if model_id is None:
        model_id = model_path.stem
    else:
        model_id = safe_model_id(model_id)
    
    print(f"\n--- Running benchmark for {model_path.name} ---")
    
    result_file = Path("results") / f"bench_{model_id}.json"
    
    cmd = [
        python_executable,
        "results/benchmark_models.py",
        "--model-path", str(model_path),
        "--model-type", model_type,
        "--batch-sizes", "1",
        "--output-json", str(result_file),
    ]
    
    try:
        subprocess.run(cmd, check=True)
        with open(result_file, 'r') as f:
            return json.load(f)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Benchmark failed: {e}")
        return None


def merge_results(model_configs: List[Dict], dataset_root: Path, python_executable: str) -> Dict:
    """
    Run eval + benchmark for each model, merge into report JSON.
    
    Args:
        model_configs: List of dicts with keys: name, path, type (family), format
        dataset_root: Path to datasets folder
    """
    models_data = []
    skipped_models = []
    failed_models = []
    
    for config in model_configs:
        if not config.get("enabled", True):
            skipped_models.append({
                "name": config.get("name", "unknown"),
                "path": str(config.get("path", "")),
                "reason": "disabled_in_config",
            })
            continue

        model_path = Path(config["path"])
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path

        model_type = config["format"]  # "pt", "onnx", "engine"
        family = config["type"]  # "teacher" or "student"
        name = config["name"]
        
        if not model_path.exists():
            print(f"[WARNING] Model not found: {model_path}")
            skipped_models.append({"name": name, "path": str(model_path), "reason": "model_not_found"})
            continue
        
        # Run evaluation
        model_id = safe_model_id(name)
        eval_result = run_evaluation(model_path, model_type, dataset_root, python_executable, model_id)
        if not eval_result:
            failed_models.append({"name": name, "path": str(model_path), "stage": "evaluation"})
            continue
        
        # Check if evaluation succeeded (at least one domain was evaluated)
        evaluated_domains = eval_result.get("evaluated_domains", 0)
        if evaluated_domains == 0:
            print(f"[WARNING] No domains evaluated for {name}. Skipping model.")
            failed_models.append({"name": name, "path": str(model_path), "stage": "evaluation", "reason": "zero_domains_evaluated"})
            continue
        
        # Run benchmark
        bench_result = run_benchmark(model_path, model_type, python_executable, model_id)
        if not bench_result:
            failed_models.append({"name": name, "path": str(model_path), "stage": "benchmark"})
            continue
        
        # Merge results
        merged = {
            "name": name,
            "family": family,
            "format": model_type,
            "metrics": eval_result.get("metrics", {}),
            "deployment": bench_result.get("deployment", {}),
            "domains": eval_result.get("domains", {}),
        }
        
        models_data.append(merged)
    
    return {
        "experiment": {
            "title": "Teacher-Student Comparison (Host Machine)",
            "hardware": "Host Machine",
            "date": Path("results").stat().st_mtime,
        },
        "models": models_data,
        "skipped_models": skipped_models,
        "failed_models": failed_models,
    }


def main():
    parser = argparse.ArgumentParser(description="Orchestrate full benchmark pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("results/model_config.json"),
        help="JSON config file with model paths and metadata"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets"),
        help="Path to datasets root"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark_metrics.json"),
        help="Output benchmark metrics JSON"
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation step"
    )
    
    args = parser.parse_args()
    
    # Load model config
    if not args.config.exists():
        print(f"Error: Config file not found at {args.config}")
        print("Create a config file with this format:")
        print(json.dumps({
            "models": [
                {"name": "Teacher_CV_pt", "path": "outputs/teachers/FASDD_CV/best.pt", "type": "teacher", "format": "pt"},
                {"name": "Student_pt", "path": "outputs/students/last.pt", "type": "student", "format": "pt"},
                {"name": "Student_engine", "path": "model_repository/swin-transform/1/model.engine", "type": "student", "format": "engine"},
            ]
        }, indent=2))
        return
    
    with args.config.open('r') as f:
        config = json.load(f)
    
    model_configs = config.get("models", [])
    if not model_configs:
        print("Error: No models defined in config")
        return
    
    print(f"{'='*70}")
    print(f"Running benchmark pipeline for {len(model_configs)} models")
    print(f"{'='*70}")
    
    # Run pipeline
    python_executable = detect_python_executable()
    print(f"Using Python interpreter: {python_executable}")

    report = merge_results(model_configs, args.dataset_root, python_executable)
    
    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"[SUCCESS] Benchmark complete!")
    print(f"[SUCCESS] Report saved: {args.output}")
    print(f"{'='*70}")
    
    if report.get("models") and not args.skip_figures:
        print(f"\nGenerating figures...")
        subprocess.run([
            python_executable,
            "results/plot_report_figures.py",
            "--input", str(args.output),
            "--output-dir", "results/figures",
        ])
    elif not report.get("models"):
        print("\n[WARNING] No successful models to plot. Check 'skipped_models' and 'failed_models' in report JSON.")


if __name__ == "__main__":
    main()
