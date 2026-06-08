## Full Benchmark Pipeline Summary

### What Changed

I created **4 new scripts** that compute real metrics from your actual models and datasets:

1. **`evaluate_models.py`** — Evaluates detection models (mAP50, Recall, F1, Precision)
   - Loads `.pt`, `.onnx`, or `.engine` models
   - Tests on all 3 domains (CV, RS, UAV)
   - Outputs per-domain + aggregate metrics

2. **`benchmark_models.py`** — Profiles model performance on host
   - Measures latency (ms), FPS, throughput per batch size
   - Tracks GPU memory usage
   - Reports model size (MB)

3. **`run_full_pipeline.py`** — Orchestrates evaluation + benchmark
   - Runs all models automatically
   - Merges results into `benchmark_metrics.json`
   - Auto-generates all 4 comparison figures

4. Plus improved versions of:
   - `plot_report_figures.py` (already working)
   - `generate_demo_benchmark.py` (already working)

### Your Evaluation Workflow

**Professional approach** (answered your second question):

- **Each teacher model** is trained on one domain (CV, RS, or UAV)
- **Each teacher is tested on ALL 3 domains** to show:
  - Domain expertise (high score on own domain)
  - Generalization (performance on other domains)
- **Student model** trained on combined data → tested on all 3 domains
- **Report metric** = macro-average (mean of CV, RS, UAV scores)

Example:
```
Teacher_CV_pt:
  - On FASDD_CV: mAP50=0.84 (high, trained here)
  - On FASDD_RS: mAP50=0.80 (generalization)
  - On FASDD_UAV: mAP50=0.77 (generalization)
  - Reported: (0.84 + 0.80 + 0.77) / 3 = 0.80
```

### Quick Start

```bash
# 1. Install deps
pip install -r results/requirements.txt

# 2. Create config with your model paths
cp results/model_config.example.json results/model_config.json
# Edit: add your actual model paths

# 3. Run everything
python3 results/run_full_pipeline.py --config results/model_config.json

# Output:
# - results/benchmark_metrics.json (merged data)
# - results/figures/figure_*.png (4 high-quality PNGs for your report)
```

### File Structure

```
results/
├── evaluate_models.py          # Compute mAP, Recall, F1
├── benchmark_models.py         # Measure latency, FPS, size
├── run_full_pipeline.py        # Orchestrator (calls both above)
├── plot_report_figures.py      # Generate PNG figures
├── generate_demo_benchmark.py  # Test data generator
│
├── model_config.example.json   # Template (copy to model_config.json)
├── benchmark_metrics.example.json # Example output format
├── README.md                   # Full documentation
├── requirements.txt            # Python dependencies
│
└── figures/                    # Output directory for PNGs
```

### Scripts Lineage

```
run_full_pipeline.py (main entry)
├── evaluate_models.py (per model, per domain) → eval_<name>.json
├── benchmark_models.py (per model) → bench_<name>.json
└── Merge both → benchmark_metrics.json
    └── plot_report_figures.py → figures/figure_*.png
```

### Next Steps

1. **Add your real model paths** to `model_config.json`
2. **Run the pipeline** (takes ~30-60 min depending on model count and dataset size)
3. **Check `results/figures/`** for your comparison charts
4. **Use PNGs in your thesis report** ✅

### Questions?

- All metrics are **per-domain + macro-averaged** (professional standard)
- Latency measured on **host machine, batch size 1** (real-time inference)
- mAP computed using **IoU threshold 0.5** (can customize in code)
- F1 is **harmonic mean of precision & recall** (standard detection metric)
