#!/usr/bin/env bash
# JETSON. Records the measurement conditions issue 1.5 asks for (module, JetPack/L4T, TensorRT, power mode,
# clock state) into results/jetson/env.txt, and optionally locks the clocks. Run it before every benchmark session
# and keep env.txt next to the results it describes.
#
#   scripts/pipeline/03_jetson_env.sh                 # record only
#   LOCK_CLOCKS=1 scripts/pipeline/03_jetson_env.sh   # also run `sudo jetson_clocks` (needs sudo)
#
# Before benchmarking: close the browser and other GUI apps (the board shares 8 GB between CPU and GPU).
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-results/jetson}"
mkdir -p "${OUT}"

if [ "${LOCK_CLOCKS:-0}" = "1" ]; then sudo jetson_clocks; fi

{
  echo "date: $(date -Is)"
  echo "module: $(tr -d '\0' < /proc/device-tree/model 2>/dev/null)"
  echo "kernel: $(uname -r)"
  echo "l4t: $(head -1 /etc/nv_tegra_release 2>/dev/null)"
  echo "nvpmodel: $( (nvpmodel -q 2>/dev/null || sudo -n nvpmodel -q 2>/dev/null) | tr '\n' ' ')"
  echo "jetson_clocks:"; (sudo -n jetson_clocks --show 2>&1 || echo "  (needs sudo; run: sudo jetson_clocks --show)") | sed 's/^/  /'
  echo "tensorrt (dpkg): $(dpkg -l 2>/dev/null | awk '/ tensorrt / {print $3; exit}')"
  echo "trtexec: $( { trtexec --version 2>&1 || true; } | grep -m1 -i 'TensorRT v' )"
  echo "deepstream: $( { deepstream-app --version-all 2>/dev/null || true; } | grep -m1 -i 'deepstream' )"
  echo "python: $(python3 --version 2>&1)"
  python3 - <<'PY' 2>/dev/null
for name in ("torch", "onnxruntime", "tensorrt", "cv2"):
    try:
        module = __import__(name)
        print(f"{name}: {module.__version__}")
    except Exception as error:
        print(f"{name}: unavailable ({type(error).__name__})")
PY
  echo "memory (MB):"; free -m | sed 's/^/  /'
  echo "tegrastats sample:"; timeout 2 tegrastats --interval 500 2>/dev/null | head -1 | sed 's/^/  /'
} | tee "${OUT}/env.txt"
echo "Saved ${OUT}/env.txt"
