#!/usr/bin/env bash
# End-to-end Pesonen et al. (PIDNet-S + SAM pseudo-masks) baseline under the shared protocol:
#   masks (once) -> train seeds -> calibrate on val -> export ONNX -> [TensorRT engine]
#   -> evaluate on test -> benchmark.
# Every stage is skipped when its output exists and is newer than its input, so the script can
# be re-run after an interruption. Training itself resumes from last.pt.
#
# Configuration (environment variables):
#   CV_BOXES, RS_BOXES, UAV_BOXES  box sources (COCO .json file(s) or YOLO labels dir); required
#                                  only while masks are incomplete. Space-separated lists allowed.
#   YOLO_CLASS_NAMES               e.g. "fire smoke" (only for YOLO dirs without classes.txt)
#   SAM_CKPT                       default third_party/weights/sam_vit_h_4b8939.pth
#   PRETRAINED                     default third_party/weights/PIDNet_S_ImageNet.pth.tar (random init if absent)
#   SEEDS                          default "42 43 44"
#   AMP                            1 (default) enables float16 autocast
#   BUILD_ENGINE                   auto (default: build if tensorrt imports) | 1 | 0
#   PYTHON                         default python
#   TRAIN_ARGS                     extra arguments for train_pidnet.py (e.g. "--val-fraction 0.25")
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-"42 43 44"}
SAM_CKPT=${SAM_CKPT:-third_party/weights/sam_vit_h_4b8939.pth}
PRETRAINED=${PRETRAINED:-third_party/weights/PIDNet_S_ImageNet.pth.tar}
AMP=${AMP:-1}
BUILD_ENGINE=${BUILD_ENGINE:-auto}
TRAIN_ARGS=${TRAIN_ARGS:-}
METHOD=pidnet_s
FAMILY=pidnet

# $1 is up to date when it exists and is newer than every other argument that exists.
fresh() {
    local target=$1; shift
    [[ -e "$target" ]] || return 1
    local src
    for src in "$@"; do
        [[ -e "$src" && "$src" -nt "$target" ]] && return 1
    done
    return 0
}

# ---- 1. SAM pseudo-masks (train + val, all domains) ----------------------------------------
if [[ -f pseudo_masks/manifest.json ]] && "$PYTHON" -c "import json,sys; sys.exit(0 if json.load(open('pseudo_masks/manifest.json')).get('complete') else 1)"; then
    echo "[masks] complete, skipping"
else
    : "${CV_BOXES:?set CV_BOXES}" "${RS_BOXES:?set RS_BOXES}" "${UAV_BOXES:?set UAV_BOXES}"
    yolo_args=()
    [[ -n "${YOLO_CLASS_NAMES:-}" ]] && yolo_args=(--yolo-class-names $YOLO_CLASS_NAMES)
    # shellcheck disable=SC2086
    "$PYTHON" scripts/pidnet/generate_sam_masks.py --sam-checkpoint "$SAM_CKPT" \
        --cv-boxes $CV_BOXES --rs-boxes $RS_BOXES --uav-boxes $UAV_BOXES "${yolo_args[@]}"
fi

engine_wanted() {
    case "$BUILD_ENGINE" in
        1) return 0 ;;
        0) return 1 ;;
        *) "$PYTHON" -c "import tensorrt" 2>/dev/null ;;
    esac
}

for seed in $SEEDS; do
    name=${METHOD}_seed${seed}
    out=outputs/baselines/${METHOD}/seed${seed}
    ckpt=$out/best.pt
    onnx=artifacts/baselines/${name}.onnx
    engine=artifacts/baselines/${name}_fp16.engine
    eval_json=results/baselines/${name}_eval.json
    bench_json=results/baselines/${name}_bench.json

    # ---- 2. train --------------------------------------------------------------------------
    if [[ -f $out/done.json ]]; then
        echo "[train] $name done, skipping"
    else
        train_args=(--seed "$seed" --output-dir "$out" --resume)
        [[ -f "$PRETRAINED" ]] && train_args+=(--pretrained "$PRETRAINED")
        [[ "$AMP" == 1 ]] && train_args+=(--amp)
        # shellcheck disable=SC2086
        "$PYTHON" scripts/pidnet/train_pidnet.py "${train_args[@]}" $TRAIN_ARGS
    fi

    # ---- 3. calibrate thresholds on val -----------------------------------------------------
    if fresh "$out/thresholds.json" "$ckpt"; then
        echo "[calibrate] $name up to date"
    else
        "$PYTHON" scripts/pidnet/calibrate_thresholds.py --checkpoint "$ckpt"
    fi

    # ---- 4. export ONNX (thresholds baked in) -----------------------------------------------
    if fresh "$onnx.manifest.json" "$ckpt" "$out/thresholds.json"; then
        echo "[export] $name up to date"
    else
        "$PYTHON" scripts/export_baseline.py --family "$FAMILY" --checkpoint "$ckpt" --output "$onnx"
    fi

    # ---- 5. optional TensorRT FP16 engine (build on the device that will run it) -------------
    engine_args=()
    if engine_wanted; then
        if fresh "$engine" "$onnx"; then
            echo "[engine] $name up to date"
        else
            "$PYTHON" scripts/build_engine.py --onnx "$onnx" --output "$engine" --precision fp16
        fi
        engine_args=(--engine "$engine")
    fi

    # ---- 6. evaluate on test (PT / ONNX / engine on identical ordered samples) ---------------
    if fresh "$eval_json" "$onnx" "${engine_args[@]:1}"; then
        echo "[evaluate] $name up to date"
    else
        "$PYTHON" scripts/evaluate_baseline.py --family "$FAMILY" --name "$name" \
            --checkpoint "$ckpt" --onnx "$onnx" "${engine_args[@]}" --output "$eval_json"
    fi

    # ---- 7. benchmark ------------------------------------------------------------------------
    if fresh "$bench_json" "$onnx" "${engine_args[@]:1}"; then
        echo "[benchmark] $name up to date"
    else
        "$PYTHON" scripts/benchmark_baseline.py --family "$FAMILY" --name "$name" \
            --checkpoint "$ckpt" --onnx "$onnx" "${engine_args[@]}" --output "$bench_json"
    fi
done
echo "All seeds finished: results/baselines/${METHOD}_seed*_{eval,bench}.json"
