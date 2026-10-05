#!/usr/bin/env bash
# JETSON. Builds TensorRT engines ON the target board from an exported ONNX (engines are not portable).
#
#   scripts/pipeline/04_build_engines.sh artifacts/retrain/kd_uniform_seed42.onnx
#   PRECISIONS="fp16 int8" CALIB_CACHE=artifacts/calibration.cache scripts/pipeline/04_build_engines.sh <onnx>
#
# Output: artifacts/engines/<onnx-name>_<precision>.engine (+ manifest with the sha256 and trtexec version).
# Max batch is 8 so the engine fits the 8 GB board; run 02_dump_predictions.py on engines with --batch-size 8 or less.
# Close the browser first: building a SwinV2-B engine needs about 2-3 GB of free memory.
set -euo pipefail
cd "$(dirname "$0")/../.."

ONNX="${1:?path to the exported .onnx (with its .manifest.json next to it)}"
NAME="$(basename "${ONNX%.onnx}")"
PRECISIONS="${PRECISIONS:-fp16}"
MAX_BATCH="${MAX_BATCH:-8}"
WORKSPACE_MIB="${WORKSPACE_MIB:-2048}"
mkdir -p artifacts/engines

for precision in ${PRECISIONS}; do
  engine="artifacts/engines/${NAME}_${precision}.engine"
  extra=()
  if [ "${precision}" = "int8" ]; then
    : "${CALIB_CACHE:?int8 needs CALIB_CACHE; create it with scripts/pipeline/04a_make_int8_calibration.py}"
    extra=(--calib-cache "${CALIB_CACHE}")
  fi
  python scripts/build_engine.py --onnx "${ONNX}" --output "${engine}" --precision "${precision}" \
    --min-batch 1 --opt-batch 1 --max-batch "${MAX_BATCH}" --workspace-mib "${WORKSPACE_MIB}" "${extra[@]}"
  echo "${engine}: $(stat -c%s "${engine}") bytes"
done
