#!/usr/bin/env bash
# MAIN MACHINE. Exports every trained student's best.pt to a manifest-verified ONNX (opset 17, input
# pixel_values, output logits). Copy the resulting artifacts/retrain/*.onnx(+.manifest.json) to the Jetson.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT_ROOT="${OUT_ROOT:-outputs/retrain}"
ART_ROOT="${ART_ROOT:-artifacts/retrain}"
SEEDS="${SEEDS:-42 43 44}"
mkdir -p "${ART_ROOT}"

for seed in ${SEEDS}; do
  for name in kd_uniform kd_routed nokd; do
    ckpt="${OUT_ROOT}/${name}/seed${seed}/best.pt"
    [ -f "${ckpt}" ] || { echo "skip ${name}/seed${seed}: no ${ckpt}"; continue; }
    python scripts/export_student.py --checkpoint "${ckpt}" --output "${ART_ROOT}/${name}_seed${seed}.onnx"
  done
done
ls -la "${ART_ROOT}"
