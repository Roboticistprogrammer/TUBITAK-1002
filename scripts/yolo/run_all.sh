#!/usr/bin/env bash
# Full YOLO baseline grid (Vazquez et al. 2025, adapted): 3 architectures x {coco, scratch} x 3 seeds.
#
#   prepare (once) -> train -> calibrate (val) -> export ONNX -> [TensorRT FP16] -> evaluate (test)
#   -> detection mAP (test) -> benchmark
#
# Resumable: every step is skipped when its final output exists (train resumes from last.pt).
# If you retrain a model, delete its downstream outputs; stale thresholds / ONNX files are
# rejected anyway (thresholds.json and the ONNX manifest are tied to the checkpoint's sha256).
#
# Usage (from anywhere; all variables are optional overrides):
#   CV_BOXES=... RS_BOXES=... UAV_BOXES=... DEVICE=0 scripts/yolo/run_all.sh
#   ARCHES="yolov8n" INITS="coco" SEEDS="42" scripts/yolo/run_all.sh     # a single cell
#   BUILD_ENGINE=1 scripts/yolo/run_all.sh                               # also build/evaluate TensorRT
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PYTHON:-python3}
ARCHES=${ARCHES:-"yolov5n yolov8n yolo11n"}
INITS=${INITS:-"coco scratch"}
SEEDS=${SEEDS:-"42 43 44"}
EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-16}
DEVICE=${DEVICE:-0}
WORKERS=${WORKERS:-8}
DATASET_ROOT=${DATASET_ROOT:-.}            # parent of datasets/
DATA_DIR=${DATA_DIR:-datasets_yolo}         # output of prepare_fasdd_yolo.py
OUT_ROOT=${OUT_ROOT:-outputs/baselines/yolo}
ART_DIR=${ART_DIR:-artifacts/baselines}
RES_DIR=${RES_DIR:-results/baselines}
BUILD_ENGINE=${BUILD_ENGINE:-0}
# Explicit optimizer so results do not depend on the Ultralytics version (docs §2.7).
TRAIN_ARGS=${TRAIN_ARGS:-"--optimizer SGD --lr0 0.01 --momentum 0.9"}
BENCH_ARGS=${BENCH_ARGS:-}                  # extra benchmark flags, e.g. "--batch-sizes 1 --runs 500"
# Box sources, only needed until $DATA_DIR/fasdd_all.yaml exists (COCO .json or YOLO labels dir).
CV_BOXES=${CV_BOXES:-}
RS_BOXES=${RS_BOXES:-}
UAV_BOXES=${UAV_BOXES:-}
YOLO_CLASS_NAMES=${YOLO_CLASS_NAMES:-}     # e.g. "fire smoke" for YOLO dirs without classes.txt

# step <marker file> <command...>: run the command unless the marker exists.
step() {
    local marker=$1
    shift
    if [[ -e "$marker" ]]; then
        echo "[skip] $marker"
    else
        echo "[run ] $*"
        "$@"
    fi
}

# 1. Dataset (once)
if [[ ! -e "$DATA_DIR/fasdd_all.yaml" ]]; then
    : "${CV_BOXES:?set CV_BOXES}" "${RS_BOXES:?set RS_BOXES}" "${UAV_BOXES:?set UAV_BOXES}"
    prepare_args=(--dataset-root "$DATASET_ROOT" --output "$DATA_DIR"
                  --cv-boxes "$CV_BOXES" --rs-boxes "$RS_BOXES" --uav-boxes "$UAV_BOXES")
    if [[ -n "$YOLO_CLASS_NAMES" ]]; then
        read -r -a class_names <<< "$YOLO_CLASS_NAMES"
        prepare_args+=(--yolo-class-names "${class_names[@]}")
    fi
    "$PY" scripts/yolo/prepare_fasdd_yolo.py "${prepare_args[@]}"
else
    echo "[skip] $DATA_DIR/fasdd_all.yaml"
fi

mkdir -p "$ART_DIR" "$RES_DIR"
for arch in $ARCHES; do
    for init in $INITS; do
        for seed in $SEEDS; do
            name="${arch}_${init}_seed${seed}"
            run="$OUT_ROOT/${arch}_${init}/seed${seed}"
            ckpt="$run/weights/best.pt"
            onnx="$ART_DIR/$name.onnx"
            engine="$ART_DIR/${name}_fp16.engine"
            echo "=== $name"

            # 2. Train (resumes an interrupted run from last.pt)
            step "$run/train_summary.json" "$PY" scripts/yolo/train_yolo.py \
                --arch "$arch" --init "$init" --seed "$seed" --epochs "$EPOCHS" --batch "$BATCH" \
                --device "$DEVICE" --workers "$WORKERS" --data "$DATA_DIR/fasdd_all.yaml" --output-root "$OUT_ROOT" $TRAIN_ARGS

            # 3. Image-level thresholds on the validation split
            step "$run/weights/thresholds.json" "$PY" scripts/yolo/calibrate_thresholds.py \
                --checkpoint "$ckpt" --dataset-root "$DATASET_ROOT"

            # 4. Verified ONNX (+ manifest; the manifest is written last, so it marks success)
            step "$onnx.manifest.json" "$PY" scripts/export_baseline.py \
                --family yolo --checkpoint "$ckpt" --output "$onnx"

            runtimes=(--onnx "$onnx")
            if [[ "$BUILD_ENGINE" == "1" ]]; then
                step "$engine" "$PY" scripts/build_engine.py --onnx "$onnx" --output "$engine" --precision fp16
                runtimes+=(--engine "$engine")
            fi

            # 5. Image-level metrics on test (PT / ONNX / [TensorRT] on identical ordered samples)
            step "$RES_DIR/${name}_eval.json" "$PY" scripts/evaluate_baseline.py \
                --family yolo --name "$name" --checkpoint "$ckpt" "${runtimes[@]}" \
                --dataset-root "$DATASET_ROOT" --output "$RES_DIR/${name}_eval.json"

            # 6. Box-level detector quality on test (secondary table)
            step "$RES_DIR/${name}_detection.json" "$PY" scripts/yolo/detection_metrics.py \
                --name "$name" --checkpoint "$ckpt" --data-dir "$DATA_DIR" --device "$DEVICE" \
                --output "$RES_DIR/${name}_detection.json"

            # 7. Latency / throughput (power on Jetson)
            step "$RES_DIR/${name}_bench.json" "$PY" scripts/benchmark_baseline.py \
                --family yolo --name "$name" --checkpoint "$ckpt" "${runtimes[@]}" \
                --output "$RES_DIR/${name}_bench.json" $BENCH_ARGS
        done
    done
done
echo "Done. Merge with: python scripts/collect_results_table.py --eval $RES_DIR/*_eval.json ... (docs/BASELINES.md)"
