#!/usr/bin/env bash
# Seeds 43/44 of the SwinV2-B students (and the no-AMP no-KD set). Resumable: a run with DONE is skipped.
# Code: worktree .claude/worktrees/students-seeds @ 56a76ef (retrain-pipeline), hyper-parameters as 00_train_students.sh
# with BATCH_SIZE=8. No-KD runs WITHOUT --amp so they match the KD students (which have no AMP path).
set -uo pipefail
MAIN=/d/Thesis-Project/TUBITAK-1002
cd "$MAIN/.claude/worktrees/students-seeds"
PY="$MAIN/.venv/Scripts/python.exe"
export PYTHONUNBUFFERED=1 PYTHONPATH=src
OUT="$MAIN/outputs/retrain"; LOGS="$OUT/logs"
T_CV="$MAIN/outputs/teachers/FASDD_CV/best.pt"; T_RS="$MAIN/outputs/teachers/FASDD_RS/best.pt"; T_UAV="$MAIN/outputs/teachers/FASDD_UAV/best.pt"

run() {  # run <dir> <command...>
  local dir="$1"; shift
  [ -f "$dir/DONE" ] && { echo "skip $dir (done)"; return 0; }
  mkdir -p "$dir"
  echo "=== $(date -Is) START $dir"
  "$@" > "$LOGS/$(basename "$(dirname "$dir")")_$(basename "$dir").log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$dir/last.pt" ]; then touch "$dir/DONE"; echo "=== $(date -Is) DONE $dir"
  else echo "=== $(date -Is) FAILED rc=$rc $dir"; exit 1; fi
}
kd() { run "$OUT/$1/seed$3" $PY scripts/train_student.py --cv-teacher "$T_CV" --rs-teacher "$T_RS" --uav-teacher "$T_UAV" \
        --aggregation "$2" --seed "$3" --epochs 20 --batch-size 8 --output-dir "$OUT/$1/seed$3"; }
nokd() { local a=(); [ -f "$OUT/nokd_noamp/seed$1/last.pt" ] && a=(--resume)
         run "$OUT/nokd_noamp/seed$1" $PY scripts/cnn/train_classifier.py --arch swinv2_base_nokd --seed "$1" --epochs 20 \
           --batch-size 8 --output-dir "$OUT/nokd_noamp/seed$1" "${a[@]}"; }

nokd 43
kd kd_routed domain_routed 44
nokd 44
kd kd_uniform weighted_avg 44
nokd 42
echo "ALL RUNS FINISHED $(date -Is)"
