#!/usr/bin/env bash
# CNN classifier baselines + no-KD SwinV2-B ablation: 3 architectures x 3 seeds.
#
#   train -> export ONNX -> [TensorRT FP16] -> evaluate (test) -> benchmark
#
# Resumable: each step is skipped when its output exists; training resumes from last.pt.
#
# Usage (all variables optional):
#   scripts/cnn/run_all.sh
#   ARCHES="resnet50" SEEDS="42" scripts/cnn/run_all.sh        # a single cell
#   BUILD_ENGINE=1 scripts/cnn/run_all.sh                       # also build/evaluate TensorRT
#   SWEEP=1 scripts/cnn/run_all.sh                              # lr sweep on seed 42 first (see docs)
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PYTHON:-python3}
ARCHES=${ARCHES:-"resnet50 mobilenet_v3_large swinv2_base_nokd"}
SEEDS=${SEEDS:-"42 43 44"}
EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-16}
WORKERS=${WORKERS:-8}
DATASET_ROOT=${DATASET_ROOT:-.}
OUT_ROOT=${OUT_ROOT:-outputs/baselines}
ART_DIR=${ART_DIR:-artifacts/baselines}
RES_DIR=${RES_DIR:-results/baselines}
BUILD_ENGINE=${BUILD_ENGINE:-0}
SWEEP=${SWEEP:-0}
SWEEP_LRS=${SWEEP_LRS:-"1e-4 3e-4 1e-3"}
LR_ARGS=${LR_ARGS:-}                     # e.g. "--lr 1e-4" after reading the sweep result
BENCH_ARGS=${BENCH_ARGS:-}

step() {
    local marker=$1
    shift
    if [[ -e "$marker" ]]; then echo "[skip] $marker"; else echo "[run ] $*"; "$@"; fi
}

family_of() { [[ "$1" == swinv2_base_nokd ]] && echo swinv2 || echo cnn; }

mkdir -p "$ART_DIR" "$RES_DIR"

# Optional tuning-parity sweep: same 3-point lr grid for every CNN, seed 42, selected on val.
if [[ "$SWEEP" == 1 ]]; then
    for arch in $ARCHES; do
        [[ "$arch" == swinv2_base_nokd ]] && continue   # uses the student's recipe verbatim
        for lr in $SWEEP_LRS; do
            step "$OUT_ROOT/sweep/${arch}_lr${lr}/best.pt" "$PY" scripts/cnn/train_classifier.py \
                --arch "$arch" --seed 42 --epochs "$EPOCHS" --batch-size "$BATCH" --num-workers "$WORKERS" \
                --lr "$lr" --amp --dataset-root "$DATASET_ROOT" --output-dir "$OUT_ROOT/sweep/${arch}_lr${lr}"
        done
    done
    "$PY" - "$OUT_ROOT/sweep" <<'EOF'
import json, sys
from pathlib import Path
for history in sorted(Path(sys.argv[1]).glob("*/history.json")):
    best = max(r["avg_val"] for r in json.loads(history.read_text())["records"])
    print(f"{history.parent.name:40s} best avg val acc = {best:.4f}")
EOF
    echo "Pick the best lr per architecture, then re-run with LR_ARGS=\"--lr <value>\" ARCHES=<arch>."
    exit 0
fi

for arch in $ARCHES; do
    family=$(family_of "$arch")
    for seed in $SEEDS; do
        name="${arch}_seed${seed}"
        run="$OUT_ROOT/$arch/seed${seed}"
        ckpt="$run/best.pt"
        onnx="$ART_DIR/$name.onnx"
        engine="$ART_DIR/${name}_fp16.engine"
        echo "=== $name"

        if [[ ! -e "$run/history.json" ]] || [[ $("$PY" -c "import json,sys;print(len(json.load(open(sys.argv[1]))['records']))" "$run/history.json") -lt "$EPOCHS" ]]; then
            "$PY" scripts/cnn/train_classifier.py --arch "$arch" --seed "$seed" --epochs "$EPOCHS" \
                --batch-size "$BATCH" --num-workers "$WORKERS" --amp --resume \
                --dataset-root "$DATASET_ROOT" --output-dir "$run" $LR_ARGS
        else
            echo "[skip] $run (training complete)"
        fi

        step "$onnx" "$PY" scripts/export_baseline.py --family "$family" --checkpoint "$ckpt" --output "$onnx"

        eval_args=(--family "$family" --name "$name" --checkpoint "$ckpt" --onnx "$onnx"
                   --dataset-root "$DATASET_ROOT" --output "$RES_DIR/${name}_eval.json")
        bench_args=(--family "$family" --name "$name" --checkpoint "$ckpt" --onnx "$onnx"
                    --output "$RES_DIR/${name}_bench.json")
        if [[ "$BUILD_ENGINE" == 1 ]]; then
            step "$engine" "$PY" scripts/build_engine.py --onnx "$onnx" --output "$engine" --precision fp16
            eval_args+=(--engine "$engine")
            bench_args+=(--engine "$engine")
        fi
        # The no-KD SwinV2 must be tagged as an ablation of the proposed method, not a literature baseline.
        [[ "$family" == swinv2 ]] && eval_args+=(--source official)

        step "$RES_DIR/${name}_eval.json" "$PY" scripts/evaluate_baseline.py "${eval_args[@]}"
        step "$RES_DIR/${name}_bench.json" "$PY" scripts/benchmark_baseline.py "${bench_args[@]}" $BENCH_ARGS
    done
done

echo "Done. Merge into the thesis table with scripts/collect_results_table.py (see docs/BASELINES.md)."
