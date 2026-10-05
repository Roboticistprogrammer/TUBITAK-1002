# Retraining and evaluation pipeline

Numbered scripts, run in order. Each says in its header which machine it is for.

| # | Script | Machine | Purpose |
|---|---|---|---|
| 00 | `00_train_students.sh` | main (GPU) | 3 seeds x {KD uniform, KD domain-routed, no-KD}, 20 epochs, teachers reused |
| 01 | `01_export_onnx.sh` | main | `best.pt` -> manifest-verified ONNX (opset 17) |
| 02 | `02_dump_predictions.py` | any | per-image logits (`.npz`) of a checkpoint, ONNX or engine on the test split |
| 03 | `03_jetson_env.sh` | Jetson | record module, L4T, TensorRT, power mode, clocks (issue 1.5); optional clock lock |
| 04 | `04_build_engines.sh` | Jetson | FP16 / INT8 TensorRT engines built on the board |
| 04a | `04a_make_int8_calibration.py` | Jetson | INT8 calibration cache from ~500-1000 **training** images |
| 05 | `05_runtime_benchmarks.sh` | Jetson | latency, peak RAM, power per runtime (PyTorch, ONNX, TRT FP16/INT8), issue 1.2 |
| 06 | `06_video_benchmark.py` | Jetson | end-to-end video FPS with decode/preprocess/inference/postprocess breakdown, issue 2.9 |
| 07 | `07_deepstream_benchmark.sh` | Jetson | DeepStream `nvinfer` throughput and per-element latency, issue 2.9 |
| 08 | `08_compute_statistics.py` | any (CPU) | per-class P/R/F1, macro vs weighted, mean +- std over seeds, bootstrap CIs, McNemar, engine-vs-PyTorch delta |

## Order of use

1. Main machine: `00` -> `01` -> `02` for every checkpoint (students, and the three teachers once, with `--checkpoint`).
2. Copy the ONNX files, the test images and a calibration subset to the Jetson. On the Jetson close the browser first
   (8 GB are shared between CPU and GPU), then `03` -> `04` -> `02` on the engines (`--batch-size 8` or less) -> `05` -> `06` -> `07`.
3. Put all `preds/*` folders on one machine and run `08`; paste `results/statistics/tables/results_summary.tex` into
   `Sections/Results.tex` (Table `tab:results-summary`) and use `summary.md` for the text.

Prediction folders are named `<group>_seed<N>[_<suffix>]`: `kd_uniform_seed42`, `kd_routed_seed42`, `nokd_seed42`,
`kd_uniform_seed42_trt_fp16`; teachers are `teacher_cv`, `teacher_rs`, `teacher_uav`. `08` pairs a deployed runtime with its
PyTorch twin by that naming.

## What was tested here (and what was not)

Tested without real data: the domain-routed loss and trainer plumbing (`tests/test_domain_routed.py`), `02` (tiny ONNX,
synthetic FASDD-shaped folders including Windows-style CSV paths), `06` (real 1080p clip, tiny ONNX, CPU), `08` (synthetic
dumps, hand-checked McNemar and metric values), `03`. **Not run:** `00`/`01` (no GPU, no checkpoints), PyTorch and TensorRT
paths of `02`/`05`/`06` (no engine yet), `04`/`04a` (INT8 calibration needs images and an engine build), and `07` with a real
engine (the board ran out of memory with the browser open). Run each once on a few images first (`02 --limit 20`,
`06 --max-frames 100`, `07` with `DECODE_ONLY=1`).
