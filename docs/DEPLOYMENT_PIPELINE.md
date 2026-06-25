# Reproducible Student Deployment Pipeline

This branch treats the Microsoft SwinV2 student as the single learned model. ONNX and
TensorRT are deployment representations of that student, not additional trained models.

## Invariants

- Architecture: `microsoft/swinv2-base-patch4-window12-192-22k`
- Classes, in order: `fire`, `smoke`, `both`, `neither`
- Model tensor: NCHW float32, `3 x 192 x 192`
- Evaluation preprocessing: resize to 224, center-crop to 192, then ImageNet normalization
- Reported data: held-out `test` split, separately for CV, RS, and UAV
- Aggregate: equal-weight mean of the three domain metrics

Every exported artifact has a sidecar manifest containing its SHA-256, source artifact
SHA-256, class order, input shape, Git commit, and tool versions. Evaluation refuses an
artifact whose manifest does not match the selected student checkpoint.

## One-time WSL GPU-container setup

The NVIDIA Container Toolkit must use WSL-aware CDI injection rather than the legacy
hook. Generate the CDI spec and configure Docker once:

```bash
sudo mkdir -p /etc/cdi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo nvidia-ctk config --in-place --set nvidia-container-runtime.mode=cdi
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU access before building the project image:

```bash
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

## Build the pinned environment

The image is based on NVIDIA PyTorch 25.08 (CUDA 13.0 generation and TensorRT 10.13).
After the first successful build, record and use the base-image digest for final thesis
runs.

```bash
docker compose build pipeline
```

## Export PT to ONNX

```bash
docker compose run --rm pipeline python scripts/export_student.py \
  --checkpoint Models/students/last.pt \
  --output artifacts/student.onnx
```

The exporter reconstructs the Microsoft model from checkpoint metadata, exports four
logits, checks the ONNX graph, and compares PyTorch and ONNX Runtime logits.

## Build TensorRT

Build the engine on the machine where it will run:

```bash
docker compose run --rm pipeline python scripts/build_engine.py \
  --onnx artifacts/student.onnx \
  --output artifacts/student_fp16.engine \
  --precision fp16
```

An engine built on the RTX 3070 is not a Jetson deployment artifact. Copy the verified
ONNX to the Jetson and rebuild the engine there in a JetPack-compatible container.

## Evaluate teachers and student runtimes

Datasets are not mounted for export or engine conversion. Attach them only when you
run evaluation. If the datasets are not located at `./datasets`, set
`DATASETS_DIR` first.

```bash
export DATASETS_DIR=/absolute/path/to/datasets

docker compose run --rm \
  --volume "${DATASETS_DIR:-./datasets}:/workspace/datasets:ro" \
  pipeline python scripts/evaluate_deployments.py \
  --student-pt Models/students/last.pt \
  --student-onnx artifacts/student.onnx \
  --student-engine artifacts/student_fp16.engine \
  --teacher-cv Models/teachers/FASDD_CV/best.pt \
  --teacher-rs Models/teachers/FASDD_RS/best.pt \
  --teacher-uav Models/teachers/FASDD_UAV/best.pt \
  --split test \
  --output results/classification_evaluation.json

docker compose run --rm pipeline python scripts/plot_confusion_matrices.py \
  --input results/classification_evaluation.json \
  --output results/confusion_matrices.png
```

The JSON contains accuracy, balanced accuracy, macro-F1 over supported classes,
per-class precision/recall/F1, raw and row-normalized confusion matrices, and exact
PT-to-ONNX/TensorRT prediction agreement.

## Benchmark deployment formats

Run every runtime in the same container and hardware session:

```bash
docker compose run --rm pipeline python scripts/benchmark_deployments.py \
  --student-pt Models/students/last.pt \
  --student-onnx artifacts/student.onnx \
  --student-engine artifacts/student_fp16.engine \
  --batch-sizes 1 4 8 \
  --output results/deployment_benchmark.json

docker compose run --rm pipeline python scripts/plot_deployment_benchmark.py \
  --input results/deployment_benchmark.json \
  --output results/deployment_benchmark.png
```

The benchmark reports mean, median, p95, standard deviation, per-image latency, and
throughput. It distinguishes the optimizer-bearing training checkpoint size from the
actual inference parameter footprint.
