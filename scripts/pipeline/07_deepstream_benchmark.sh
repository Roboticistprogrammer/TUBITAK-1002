#!/usr/bin/env bash
# JETSON. DeepStream (nvinfer) throughput and per-element latency on a video, for issue 2.9. Triton is not benchmarked.
# H.264 .mp4 input only (qtdemux ! h264parse ! nvv4l2decoder). Run 03_jetson_env.sh first and keep one stack running at a time.
#
#   scripts/pipeline/07_deepstream_benchmark.sh artifacts/engines/kd_uniform_seed42_fp16.engine videos/flame2_fhd.mp4 [batch]
#   DECODE_ONLY=1 scripts/pipeline/07_deepstream_benchmark.sh - videos/flame2_fhd.mp4   # decode + mux only: pipeline cost without the model
#   DRY_RUN=1 ...                                                                       # print the pipeline and config, run nothing
#
# Output: results/jetson/deepstream/<engine>_<video>_b<batch>/{nvinfer.txt,gst.log,tegrastats.log,summary.json}.
# summary.json: frames, seconds, end-to-end FPS, mean per-element latency in ms (GStreamer latency tracer), power and RAM.
# Differences from the PyTorch/ORT/TensorRT harness: nvinfer resizes the whole frame straight to 192x192 on the GPU
# (no 224 resize + centre crop) and applies one scalar scale factor (std ~0.226 instead of per-channel std), so this
# measures pipeline throughput, not accuracy. Untested until a real engine exists; check gst.log if nvinfer rebuilds the engine.
set -euo pipefail
cd "$(dirname "$0")/../.."

ENGINE="${1:?engine path (or - with DECODE_ONLY=1)}"
VIDEO="${2:?video path (H.264 mp4)}"
BATCH="${3:-1}"
DECODE_ONLY="${DECODE_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
TAG="$(basename "${ENGINE%.engine}")_$(basename "${VIDEO%.*}")_b${BATCH}"
[ "${DECODE_ONLY}" = "1" ] && TAG="decode_only_$(basename "${VIDEO%.*}")_b${BATCH}"
OUT="results/jetson/deepstream/${TAG}"
mkdir -p "${OUT}"

read -r WIDTH HEIGHT < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "${VIDEO}" | tr ',' ' ')
FRAMES="$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "${VIDEO}")"

printf 'fire\nsmoke\nboth\nneither\n' > "${OUT}/labels.txt"
MODE=2; case "${ENGINE}" in *int8*) MODE=1;; *fp32*) MODE=0;; esac
cat > "${OUT}/nvinfer.txt" <<EOF
[property]
gpu-id=0
net-scale-factor=0.01735
offsets=123.675;116.28;103.53
model-color-format=0
model-engine-file=$(realpath -m "${ENGINE}")
labelfile-path=$(realpath "${OUT}/labels.txt")
batch-size=${BATCH}
network-mode=${MODE}
process-mode=1
network-type=1
num-detected-classes=4
interval=0
gie-unique-id=1
infer-dims=3;192;192
classifier-threshold=0
EOF

if [ "${DECODE_ONLY}" = "1" ]; then
  INFER="identity"
else
  INFER="nvinfer config-file-path=$(realpath "${OUT}/nvinfer.txt")"
fi
PIPELINE="filesrc location=${VIDEO} ! qtdemux ! h264parse ! nvv4l2decoder ! mux.sink_0 nvstreammux name=mux batch-size=${BATCH} width=${WIDTH} height=${HEIGHT} batched-push-timeout=40000 ! ${INFER} ! fakesink sync=false"
echo "frames=${FRAMES} size=${WIDTH}x${HEIGHT}"; echo "gst-launch-1.0 -e ${PIPELINE}"
[ "${DRY_RUN}" = "1" ] && exit 0

tegrastats --interval 200 > "${OUT}/tegrastats.log" &
TEGRA=$!
trap 'kill ${TEGRA} 2>/dev/null || true' EXIT
START="$(date +%s.%N)"
if ! GST_TRACERS="latency(flags=element)" GST_DEBUG="GST_TRACER:7" \
  gst-launch-1.0 -e ${PIPELINE} > "${OUT}/gst.log" 2>&1; then
  echo "gst-launch failed. Last lines of ${OUT}/gst.log:" >&2
  grep -v GST_TRACER "${OUT}/gst.log" | tail -8 >&2
  echo "NvMapMemAlloc errors mean the board is out of memory: close the browser/GUI apps and retry." >&2
  exit 1
fi
END="$(date +%s.%N)"
kill "${TEGRA}" 2>/dev/null || true

PYTHONPATH=src python3 - "${OUT}" "${FRAMES}" "${START}" "${END}" "${BATCH}" "${WIDTH}x${HEIGHT}" "${ENGINE}" <<'PY'
import json, re, sys
from collections import defaultdict
from pathlib import Path
from firecls.baselines.power import parse_power_mw, parse_ram_mb

out, frames, start, end, batch, size, engine = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
latency = defaultdict(list)
pattern = re.compile(r"element-latency.*?element=\(string\)(?P<element>[^,]+),.*?time=\(guint64\)(?P<time>\d+)")
for line in (out / "gst.log").read_text(errors="ignore").splitlines():
    match = pattern.search(line)
    if match:
        latency[match.group("element")].append(int(match.group("time")) / 1e6)
power, ram = [], []
for line in (out / "tegrastats.log").read_text(errors="ignore").splitlines():
    if (p := parse_power_mw(line)): power.append(p[1])
    if (r := parse_ram_mb(line)): ram.append(r[0])
seconds = end - start
summary = {
    "engine": engine, "frames": frames, "resolution": size, "batch_size": batch, "wall_seconds": seconds,
    "end_to_end_fps": frames / seconds,
    "element_latency_ms_mean": {k: sum(v) / len(v) for k, v in latency.items()},
    "power_mean_mw": sum(power) / len(power) if power else None, "power_max_mw": max(power) if power else None,
    "peak_ram_mb": max(ram) if ram else None,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
