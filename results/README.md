# Results Figure Pipeline

This folder generates **report-ready figures** (not CSV tables) for comparing Teacher vs. Student models.

## Professional Evaluation Protocol

**Per-Domain Testing**: Each model is tested on **all three domains** (FASDD_CV, FASDD_RS, FASDD_UAV):
- Teacher models (trained per-domain) → tested on CV/RS/UAV to show domain expertise + generalization
- Student model (trained on all domains) → tested on each domain individually
- **Reported metrics** = macro-average across domains (fair comparison)

## Full Pipeline (Real Data)

**`run_full_pipeline.py`** — Orchestrates everything in one command:
1. Evaluates each model on all three domains (computes mAP50, Recall, F1)
2. Benchmarks each model for latency, FPS, model size
3. Merges results into `benchmark_metrics.json`
4. Generates all figures

**Quick Start:**
```bash
# 1. Create model config
cp results/model_config.example.json results/model_config.json
# Edit with your model paths

# 2. Run full pipeline
python3 -m pip install -r results/requirements.txt
python3 results/run_full_pipeline.py --config results/model_config.json
```

## Individual Scripts

- **`evaluate_models.py`** — Compute mAP, Recall, F1 per model on all domains
- **`benchmark_models.py`** — Measure latency, FPS, memory on host machine
- **`plot_report_figures.py`** — Generate colorful comparison figures
- **`generate_demo_benchmark.py`** — Create sample data for testing

## Model Config Format

Create `results/model_config.json`:

```json
{
  "models": [
    {"name": "Teacher_CV_pt", "path": "outputs/teachers/FASDD_CV/best.pt", "type": "teacher", "format": "pt"},
    {"name": "Teacher_RS_pt", "path": "outputs/teachers/FASDD_RS/best.pt", "type": "teacher", "format": "pt"},
    {"name": "Teacher_UAV_pt", "path": "outputs/teachers/FASDD_UAV/best.pt", "type": "teacher", "format": "pt"},
    {"name": "Student_pt", "path": "outputs/students/last.pt", "type": "student", "format": "pt"},
    {"name": "Student_onnx", "path": "model_repository/swin-transform/1/model.onnx", "type": "student", "format": "onnx"},
    {"name": "Student_engine", "path": "model_repository/swin-transform/1/model.engine", "type": "student", "format": "engine"}
  ]
}
```

## Output Figures

- `figure_accuracy_comparison.png` — mAP50, Recall, F1 bars
- `figure_domain_demographics.png` — Per-domain mAP50 breakdown
- `figure_deployment_comparison.png` — Latency, FPS, model size
- `figure_dashboard.png` — 2x2 summary grid
