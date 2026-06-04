# Jetson Benchmark + Visual Report Workflow

This repository includes two dedicated scripts for testing your models separately on Jetson:

- `benchmark_models.py`: numeric performance benchmarking
- `visualize_outputs.py`: report-ready visual predictions

## Models

Current models:

- `models/model.onnx`
- `models/model.engine`

## Metrics Captured

`benchmark_models.py` captures:

- Latency (mean, p50, p95)
- Throughput (images/sec)
- FPS
- Power (avg/max watts from `tegrastats`)
- Energy per image (J)

## Install

```bash
python3 -m pip install -r requirements.txt
```

Jetson notes:

- `tensorrt` and `pycuda` typically come from JetPack/NVIDIA packages.
- For ONNX GPU, use your Jetson-compatible `onnxruntime-gpu` build.

## 1) Benchmark ONNX model

```bash
python3 benchmark_models.py \
	--model-path models/model.onnx \
	--model-type onnx \
	--sample-dir sample \
	--batch-size 1 \
	--warmup 20 \
	--runs 100
```

## 2) Benchmark TensorRT engine model

```bash
python3 benchmark_models.py \
	--model-path models/model.engine \
	--model-type engine \
	--sample-dir sample \
	--batch-size 1 \
	--warmup 20 \
	--runs 100
```

## 3) Generate visual outputs for ONNX model

```bash
python3 visualize_outputs.py \
	--model-path models/model.onnx \
	--model-type onnx \
	--sample-dir sample \
	--class-names fire,neither,smoke \
	--max-images-per-class 30 \
	--save-contact-sheet
```

## 4) Generate visual outputs for TensorRT engine

```bash
python3 visualize_outputs.py \
	--model-path models/model.engine \
	--model-type engine \
	--sample-dir sample \
	--class-names fire,neither,smoke \
	--max-images-per-class 30 \
	--save-contact-sheet
```

## Output Structure

- Benchmarks: `outputs/benchmarks/<model_name>/<timestamp>/`
	- `metrics.json`
	- `metrics.csv`
	- `tegrastats.log`
- Visuals: `outputs/visuals/<model_name>/<timestamp>/`
	- annotated images under original `sample/` hierarchy
	- `predictions.json`
	- `predictions.csv`
	- `contact_sheet.png` (if enabled)

## Existing Triton Flow

If needed, the existing Triton setup remains available via `triton_client.py`.