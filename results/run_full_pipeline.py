"""
Orchestrate full evaluation pipeline:
  1. Run evaluate_models.py on each model (.pt, .onnx, .engine)
  2. Run benchmark_models.py on each model
  3. Merge results into benchmark_metrics.json for plotting
"""

import argparse
import json
import platform
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
    def has_ultralytics(executable: str) -> bool:
        try:
            check = subprocess.run(
                [executable, "-c", "import ultralytics"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return check.returncode == 0
        except Exception:
            return False

    on_linux = platform.system() == "Linux"

    candidates = [
        Path(sys.executable) if sys.executable else None,
        Path.cwd() / ".venv" / "bin" / "python",
    ]
    # Windows venv exe is only valid when actually running on Windows, not WSL
    if not on_linux:
        candidates.append(Path.cwd() / ".venv" / "Scripts" / "python.exe")

    for candidate in candidates:
        if candidate and candidate.exists() and has_ultralytics(str(candidate)):
            return str(candidate)
    return sys.executable


def is_windows_python(python_executable: str) -> bool:
    return python_executable.lower().endswith(".exe")


def to_subprocess_path(path: Path, python_executable: str) -> str:
    resolved = path.resolve()
    path_str = str(resolved)

    if is_windows_python(python_executable) and path_str.startswith("/mnt/") and len(path_str) > 6:
        drive_letter = path_str[5].upper()
        remainder = path_str[6:].replace("/", "\\")
        return f"{drive_letter}:{remainder}"

    return path_str


def run_evaluation(model_path: Path, model_type: str, dataset_root: Path, python_executable: str, model_id: str = None) -> Dict:
    """Run evaluation script and return results."""
    if model_id is None:
        model_id = model_path.stem
    else:
        model_id = safe_model_id(model_id)
    
    print(f"\n--- Running evaluation for {model_path.name} ---")
    
    result_file = (Path.cwd() / "results" / f"eval_{model_id}.json").resolve()
    evaluate_script = (Path.cwd() / "results" / "evaluate_models.py").resolve()
    
    cmd = [
        python_executable,
        to_subprocess_path(evaluate_script, python_executable),
        "--model-path", to_subprocess_path(model_path, python_executable),
        "--model-type", model_type,
        "--dataset-root", to_subprocess_path(dataset_root, python_executable),
        "--output-json", to_subprocess_path(result_file, python_executable),
    ]
    
    try:
        subprocess.run(cmd, check=True)
        if not result_file.exists():
            print(f"[ERROR] Evaluation did not produce output file: {result_file}")
            return None
        with open(result_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Evaluation output file missing: {result_file}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Evaluation failed: {e}")
        return None


def run_benchmark(model_path: Path, model_type: str, python_executable: str, model_id: str = None, imgsz: int = 640) -> Dict:
    """Run benchmark script and return results."""
    if model_id is None:
        model_id = model_path.stem
    else:
        model_id = safe_model_id(model_id)
    
    print(f"\n--- Running benchmark for {model_path.name} ---")
    
    result_file = (Path.cwd() / "results" / f"bench_{model_id}.json").resolve()
    benchmark_script = (Path.cwd() / "results" / "Benchmark_Engine.py").resolve()
    
    cmd = [
        python_executable,
        to_subprocess_path(benchmark_script, python_executable),
        "--model-path", to_subprocess_path(model_path, python_executable),
        "--model-type", model_type,
        "--batch-sizes", "1", "4", "8",
        "--output-json", to_subprocess_path(result_file, python_executable),
    ]
    if imgsz and imgsz != 640:
        cmd += ["--imgsz", str(imgsz)]
    
    try:
        subprocess.run(cmd, check=True)
        if not result_file.exists():
            print(f"[ERROR] Benchmark did not produce output file: {result_file}")
            return None
        with open(result_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Benchmark output file missing: {result_file}")
        return None
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
        
        # Optionally skip evaluation (deployment-only comparison)
        model_id = safe_model_id(name)
        skip_eval = config.get("skip_evaluation", False)

        if skip_eval:
            print(f"  [INFO] Skipping evaluation for {name} (skip_evaluation=true in config)")
            eval_result = {"metrics": {}, "domains": {}, "evaluated_domains": 1}
        else:
            eval_result = run_evaluation(model_path, model_type, dataset_root, python_executable, model_id)
            if not eval_result:
                failed_models.append({"name": name, "path": str(model_path), "stage": "evaluation"})
                continue

            evaluated_domains = eval_result.get("evaluated_domains", 0)
            if evaluated_domains == 0:
                print(f"[WARNING] No domains evaluated for {name}. Skipping model.")
                failed_models.append({"name": name, "path": str(model_path), "stage": "evaluation", "reason": "zero_domains_evaluated"})
                continue

        # Run benchmark
        bench_result = run_benchmark(model_path, model_type, python_executable, model_id, imgsz=config.get("imgsz", 640))
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
                {"name": "Teacher_CV_pt", "path": "Models/teachers/FASDD_CV/best.pt", "type": "teacher", "format": "pt"},
                {"name": "Student_pt", "path": "Models/students/last.pt", "type": "student", "format": "pt"},
                {"name": "Student_engine", "path": "Models/model.engine", "type": "student", "format": "engine"},
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
        figures_script = (Path.cwd() / "results" / "plot_report_figures.py").resolve()
        figures_output = (Path.cwd() / "results" / "figures").resolve()
        subprocess.run([
            python_executable,
            to_subprocess_path(figures_script, python_executable),
            "--input", to_subprocess_path(args.output.resolve(), python_executable),
            "--output-dir", to_subprocess_path(figures_output, python_executable),
        ])
    elif not report.get("models"):
        print("\n[WARNING] No successful models to plot. Check 'skipped_models' and 'failed_models' in report JSON.")


if __name__ == "__main__":
    main()
