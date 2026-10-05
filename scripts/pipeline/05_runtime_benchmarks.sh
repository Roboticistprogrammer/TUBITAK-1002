#!/usr/bin/env bash
# JETSON. Latency, throughput, peak RAM and power for each runtime of ONE model, one runtime per process so the
# memory numbers are not contaminated by an earlier runtime (issues 1.2 and 1.5). Run 03_jetson_env.sh first.
#
#   NAME=kd_uniform_seed42 CKPT=outputs/retrain/kd_uniform/seed42/best.pt \
#   ONNX=artifacts/retrain/kd_uniform_seed42.onnx \
#   ENGINE_FP16=artifacts/engines/kd_uniform_seed42_fp16.engine ENGINE_INT8=artifacts/engines/kd_uniform_seed42_int8.engine \
#   scripts/pipeline/05_runtime_benchmarks.sh
#
# Any of CKPT / ONNX / ENGINE_FP16 / ENGINE_INT8 may be omitted. Output: results/jetson/<NAME>_<runtime>.json with
# latency percentiles per batch size, `memory` (peak_ram_mb; footprint = peak_ram_mb - environment.idle_ram_used_mb)
# and `power`/`energy_mj_per_image`. The ONNX Runtime row uses CUDA only if onnxruntime-gpu is installed; the JSON's
# artifact record does not say which provider ran, so check `python -c "import onnxruntime as o;print(o.get_available_providers())"`
# and label the row "ORT CPU" if CUDAExecutionProvider is absent.
set -euo pipefail
cd "$(dirname "$0")/../.."

NAME="${NAME:?set NAME, e.g. kd_uniform_seed42}"
OUT="${OUT:-results/jetson}"
BATCHES="${BATCHES:-1 4 8}"
RUNS="${RUNS:-200}"
COOLDOWN="${COOLDOWN:-20}"
mkdir -p "${OUT}"

bench() {  # bench <runtime-label> <flag> <path>
  local label="$1" flag="$2" path="$3"
  [ -n "${path}" ] || return 0
  echo "=== ${NAME} ${label}"
  python scripts/benchmark_baseline.py --family swinv2 --name "${NAME}_${label}" "${flag}" "${path}" \
    --batch-sizes ${BATCHES} --runs "${RUNS}" --output "${OUT}/${NAME}_${label}.json"
  sleep "${COOLDOWN}"  # let the board cool so thermal throttling does not bias the next runtime
}

bench pytorch   --checkpoint "${CKPT:-}"
bench onnx      --onnx       "${ONNX:-}"
bench trt_fp16  --engine     "${ENGINE_FP16:-}"
bench trt_int8  --engine     "${ENGINE_INT8:-}"
echo "Results in ${OUT}/${NAME}_*.json"
