#!/usr/bin/env bash
# MAIN MACHINE (CUDA GPU). Trains every student needed for the thesis results:
#   3 seeds x { KD with uniform teacher aggregation, KD domain-routed, no-KD hard-label baseline }.
# The three SwinV2-L teachers are reused as they are (they are not retrained).
#
#   TEACHER_CV=... TEACHER_RS=... TEACHER_UAV=... scripts/pipeline/00_train_students.sh
#
# Run it under tmux/nohup and DO NOT interrupt: the previous student stopped at epoch 11 of 20 (issue 2.11).
# A run that finished is marked with <run-dir>/DONE and skipped on re-launch; an interrupted KD run restarts from scratch.
set -euo pipefail
cd "$(dirname "$0")/../.."

TEACHER_CV="${TEACHER_CV:-outputs/teachers/FASDD_CV/best.pt}"
TEACHER_RS="${TEACHER_RS:-outputs/teachers/FASDD_RS/best.pt}"
TEACHER_UAV="${TEACHER_UAV:-outputs/teachers/FASDD_UAV/best.pt}"
SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"          # protocol value; lower it only if the GPU runs out of memory, and say so in the thesis
OUT_ROOT="${OUT_ROOT:-outputs/retrain}"
mkdir -p "${OUT_ROOT}/logs"

for f in "$TEACHER_CV" "$TEACHER_RS" "$TEACHER_UAV"; do
  [ -f "$f" ] || { echo "missing teacher checkpoint: $f" >&2; exit 1; }
done

run() {  # run <run-dir> <command...>
  local dir="$1"; shift
  if [ -f "${dir}/DONE" ]; then echo "skip ${dir} (done)"; return; fi
  mkdir -p "${dir}"
  echo "=== $(date -Is) ${dir}"
  "$@" 2>&1 | tee "${OUT_ROOT}/logs/$(echo "${dir}" | tr '/' '_').log"
  touch "${dir}/DONE"
}

for seed in ${SEEDS}; do
  for variant in "kd_uniform:weighted_avg" "kd_routed:domain_routed"; do
    name="${variant%%:*}"; aggregation="${variant##*:}"
    run "${OUT_ROOT}/${name}/seed${seed}" \
      python scripts/train_student.py \
        --cv-teacher "$TEACHER_CV" --rs-teacher "$TEACHER_RS" --uav-teacher "$TEACHER_UAV" \
        --aggregation "${aggregation}" --seed "${seed}" --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" \
        --output-dir "${OUT_ROOT}/${name}/seed${seed}"
  done
  run "${OUT_ROOT}/nokd/seed${seed}" \
    python scripts/cnn/train_classifier.py --arch swinv2_base_nokd --seed "${seed}" \
      --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --amp --output-dir "${OUT_ROOT}/nokd/seed${seed}"
done
echo "All runs finished. Checkpoints: ${OUT_ROOT}/<kd_uniform|kd_routed|nokd>/seed<N>/best.pt"
