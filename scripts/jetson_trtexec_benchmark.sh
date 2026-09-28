#!/usr/bin/env bash
# Version-agnostic Jetson benchmark: builds an FP16 engine ON the device and times it with
# trtexec while logging tegrastats. Works on every JetPack (including JetPack 4 / TensorRT 8.2
# on Jetson Nano, whose Python 3.6 cannot run the repository's Python tooling).
#
# Usage:  scripts/jetson_trtexec_benchmark.sh artifacts/baselines/resnet50_seed42.onnx 192 [batch]
#
# Record for the thesis: `cat /etc/nv_tegra_release`, `sudo nvpmodel -q`, and that
# `sudo jetson_clocks` was applied before running.
set -euo pipefail

ONNX="${1:?path to .onnx}"
SIZE="${2:?input side length, e.g. 192 or 640}"
BATCH="${3:-1}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
NAME="$(basename "${ONNX%.onnx}")"
OUT="results/jetson/${NAME}_b${BATCH}"
mkdir -p "${OUT}"

SHAPE="pixel_values:${BATCH}x3x${SIZE}x${SIZE}"
ENGINE="${OUT}/${NAME}_fp16_b${BATCH}.engine"

"${TRTEXEC}" --onnx="${ONNX}" --saveEngine="${ENGINE}" --fp16 \
  --minShapes="${SHAPE}" --optShapes="${SHAPE}" --maxShapes="${SHAPE}" \
  --workspace=1024 > "${OUT}/build.log" 2>&1 \
  || "${TRTEXEC}" --onnx="${ONNX}" --saveEngine="${ENGINE}" --fp16 \
       --minShapes="${SHAPE}" --optShapes="${SHAPE}" --maxShapes="${SHAPE}" \
       --memPoolSize=workspace:1024 > "${OUT}/build.log" 2>&1   # TensorRT >= 8.4 flag

tegrastats --interval 100 > "${OUT}/tegrastats.log" &
TEGRA_PID=$!
trap 'kill ${TEGRA_PID} 2>/dev/null || true' EXIT

"${TRTEXEC}" --loadEngine="${ENGINE}" --shapes="${SHAPE}" \
  --warmUp=2000 --duration=30 --iterations=500 --percentile=99 \
  --exportTimes="${OUT}/times.json" > "${OUT}/timing.log" 2>&1

kill ${TEGRA_PID} 2>/dev/null || true
grep -E "Throughput|Latency|GPU Compute" "${OUT}/timing.log" | tee "${OUT}/summary.txt"
echo "Logs in ${OUT}. Summarise on any machine with Python >= 3.8:"
echo "  python scripts/summarize_jetson_run.py ${OUT} --batch ${BATCH} --name ${NAME}"
