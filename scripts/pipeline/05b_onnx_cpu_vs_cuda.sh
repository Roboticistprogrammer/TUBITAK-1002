#!/usr/bin/env bash
# JETSON. Latency/throughput/RAM/power of ONNX Runtime on CPU vs CUDA, plus the TensorRT FP16 engine, for each model.
# Needs the CUDA onnxruntime build (e.g. ~/venvs-firecls-gpu: uv venv --system-site-packages, then
# uv pip install --no-deps --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple onnxruntime-gpu==1.23.0).
# ENGINE_BATCHES defaults to 1 because the SwinV2 engines were built with a batch-1 profile; use "1 4 8" for engines built
# with --max-batch 8. ONNX always runs batch 1 4 8. Model <m> is <ONNX_DIR>/<m>_seed42.onnx and <ENGINE_DIR>/<m>_seed42_fp16.engine.
#
#   PY=~/venvs-firecls-gpu/bin/python MODELS="kd_routed kd_uniform nokd" scripts/pipeline/05b_onnx_cpu_vs_cuda.sh
#   PY=... MODELS=mobilenet_v3_large ONNX_DIR=<dir with mobilenet_v3_large_seed42.onnx> ENGINE_BATCHES="1 4 8" scripts/pipeline/05b_onnx_cpu_vs_cuda.sh
# Output: results/jetson/<model>_seed42_{ort_cpu,ort_cuda,trt_fp16}.json (provider recorded in the JSON).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"; OUT="${OUT:-results/jetson}"; RUNS="${RUNS:-100}"; COOLDOWN="${COOLDOWN:-20}"
MODELS="${MODELS:-kd_routed kd_uniform nokd}"
ONNX_DIR="${ONNX_DIR:-artifacts/retrain}"; ENGINE_DIR="${ENGINE_DIR:-artifacts/engines}"; ENGINE_BATCHES="${ENGINE_BATCHES:-1}"
FAMILY="${FAMILY:-swinv2}"  # only used by benchmark_baseline.py to load checkpoints; irrelevant for ONNX/engines
mkdir -p "$OUT"

# Jetson quirk: pip torch (~/.local/.../nvidia/*) loads its own cuBLAS/cuDNN first, and onnxruntime-gpu then fails with
# CUBLAS_STATUS_ALLOC_FAILED / CUDNN_BACKEND_API_FAILED. Preloading the system JetPack libs makes both use the same copies.
CUDA_LIB=/usr/local/cuda/targets/aarch64-linux/lib; SYS_LIB=/usr/lib/aarch64-linux-gnu
ORT_CUDA_PRELOAD="$CUDA_LIB/libcublasLt.so.12 $CUDA_LIB/libcublas.so.12 $SYS_LIB/libcudnn.so.9"
for c in graph engines_precompiled engines_runtime_compiled ops cnn adv heuristic; do ORT_CUDA_PRELOAD+=" $SYS_LIB/libcudnn_$c.so.9"; done
for m in $MODELS; do
  n="${m}_seed42"
  "$PY" scripts/benchmark_baseline.py --family "$FAMILY" --name "${n}_ort_cpu" --onnx "$ONNX_DIR/${n}.onnx" --onnx-cpu \
    --batch-sizes 1 4 8 --runs "$RUNS" --output "$OUT/${n}_ort_cpu.json"; sleep "$COOLDOWN"
  LD_LIBRARY_PATH="$SYS_LIB:$CUDA_LIB:${LD_LIBRARY_PATH:-}" LD_PRELOAD="$ORT_CUDA_PRELOAD" \
  "$PY" scripts/benchmark_baseline.py --family "$FAMILY" --name "${n}_ort_cuda" --onnx "$ONNX_DIR/${n}.onnx" \
    --batch-sizes 1 4 8 --runs "$RUNS" --output "$OUT/${n}_ort_cuda.json"; sleep "$COOLDOWN"
  "$PY" scripts/benchmark_baseline.py --family "$FAMILY" --name "${n}_trt_fp16" --engine "$ENGINE_DIR/${n}_fp16.engine" \
    --batch-sizes $ENGINE_BATCHES --runs "$RUNS" --output "$OUT/${n}_trt_fp16.json"; sleep "$COOLDOWN"
done
echo "Results in $OUT"
