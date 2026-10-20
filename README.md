# Test-dataset results: ONNX and TensorRT engines

Results of the exported **ONNX** and **TensorRT FP16 engine** models on the held-out FASDD fire/smoke test split (`amir-salar/test-dataset`: 20,435 images, CV 15,884 / UAV 4,181 / RS 370; classes `fire`, `smoke`, `both`, `neither`). All models are seed 42. The weights, ONNX files and engines are not stored here (size; engines are board-specific).

Produced on the `retrain-pipeline` branch: `scripts/pipeline/02_dump_predictions.py` (accuracy), `scripts/pipeline/05b_onnx_cpu_vs_cuda.sh` (latency), `scripts/summarize_test_dataset.py` and `scripts/summarize_jetson_benchmarks.py` (tables).

## Layout

```
<architecture>/
  README.md          accuracy + latency tables for that architecture
  predictions/<model>_{onnx,trt_fp16}/{cv,uav,rs}.npz, meta.json   per-image logits, labels, file names
  benchmarks/<model>_{ort_cpu,ort_cuda,trt_fp16}.json              latency percentiles, power, RAM per batch size
accuracy_summary.csv   latency_throughput.csv   (all architectures)
```

## SwinV2-B students (87M params; KD routed, KD uniform, no-KD)

### Accuracy

| model | runtime | CV acc | UAV acc | RS acc | CV macro-F1 | UAV macro-F1 | RS macro-F1 |
|---|---|---|---|---|---|---|---|
| kd_routed_seed42 | ONNX Runtime (CPU) | 0.9182 | 0.9416 | 0.9000 | 0.9029 | 0.8948 | 0.6075 |
| kd_routed_seed42 | TensorRT FP16 | 0.9181 | 0.9416 | 0.9000 | 0.9028 | 0.8948 | 0.6075 |
| kd_uniform_seed42 | ONNX Runtime (CPU) | 0.9168 | 0.9514 | 0.8703 | 0.9047 | 0.9134 | 0.5986 |
| kd_uniform_seed42 | TensorRT FP16 | 0.9169 | 0.9519 | 0.8703 | 0.9047 | 0.9138 | 0.5986 |
| nokd_seed42 | ONNX Runtime (CPU) | 0.9105 | 0.9388 | 0.8838 | 0.8997 | 0.9015 | 0.6163 |
| nokd_seed42 | TensorRT FP16 | 0.9104 | 0.9390 | 0.8865 | 0.8996 | 0.9017 | 0.6177 |

### Latency / throughput (Jetson Orin, random 192x192 input, preprocessing excluded)

| model | runtime | batch | p50 ms/batch | p95 ms/batch | img/s | mean power mW | mJ/img | peak RAM MB |
|---|---|---|---|---|---|---|---|---|
| kd_routed_seed42 | ORT CPU | 1 | 1022.3 | 1333.8 | 1.0 | 8202 | 8049 | 4514 |
| kd_routed_seed42 | ORT CPU | 4 | 1798.7 | 2440.1 | 2.3 | 8749 | 3752 | 4520 |
| kd_routed_seed42 | ORT CPU | 8 | 2201.4 | 2565.2 | 3.5 | 9094 | 2585 | 4462 |
| kd_routed_seed42 | ORT CUDA | 1 | 34.9 | 36.1 | 28.5 | 15776 | 554 | 4595 |
| kd_routed_seed42 | ORT CUDA | 4 | 91.1 | 91.6 | 43.9 | 18697 | 426 | 4618 |
| kd_routed_seed42 | ORT CUDA | 8 | 165.5 | 169.1 | 48.2 | 19826 | 412 | 4894 |
| kd_routed_seed42 | TensorRT FP16 | 1 | 11.8 | 14.7 | 81.1 | 12727 | 157 | 4468 |
| kd_uniform_seed42 | ORT CPU | 1 | 331.2 | 1014.1 | 1.9 | 7913 | 4089 | 4439 |
| kd_uniform_seed42 | ORT CPU | 4 | 1129.3 | 1283.3 | 3.5 | 8447 | 2432 | 4376 |
| kd_uniform_seed42 | ORT CPU | 8 | 2226.7 | 2328.2 | 3.6 | 8549 | 2397 | 4499 |
| kd_uniform_seed42 | ORT CUDA | 1 | 34.6 | 35.0 | 28.7 | 15851 | 553 | 4539 |
| kd_uniform_seed42 | ORT CUDA | 4 | 90.8 | 91.1 | 44.0 | 18724 | 425 | 4805 |
| kd_uniform_seed42 | ORT CUDA | 8 | 164.3 | 165.3 | 48.7 | 19595 | 402 | 4803 |
| kd_uniform_seed42 | TensorRT FP16 | 1 | 11.9 | 17.1 | 78.9 | 12080 | 153 | 4399 |
| nokd_seed42 | ORT CPU | 1 | 322.3 | 349.1 | 3.1 | 8298 | 2700 | 4311 |
| nokd_seed42 | ORT CPU | 4 | 1123.2 | 1269.8 | 3.3 | 8316 | 2520 | 4359 |
| nokd_seed42 | ORT CPU | 8 | 2195.9 | 2243.4 | 3.6 | 8552 | 2352 | 4463 |
| nokd_seed42 | ORT CUDA | 1 | 34.7 | 35.3 | 28.7 | 15876 | 553 | 4552 |
| nokd_seed42 | ORT CUDA | 4 | 90.6 | 90.9 | 44.1 | 18843 | 427 | 4561 |
| nokd_seed42 | ORT CUDA | 8 | 164.6 | 165.6 | 48.6 | 19615 | 404 | 4857 |
| nokd_seed42 | TensorRT FP16 | 1 | 11.8 | 17.1 | 80.5 | 12374 | 154 | 4461 |

## MobileNetV3-Large baseline (4.2M params)

### Accuracy

| model | runtime | CV acc | UAV acc | RS acc | CV macro-F1 | UAV macro-F1 | RS macro-F1 |
|---|---|---|---|---|---|---|---|
| mobilenet_v3_large_seed42 | ONNX Runtime (CPU) | 0.9117 | 0.9522 | 0.8649 | 0.8980 | 0.9243 | 0.5882 |
| mobilenet_v3_large_seed42 | TensorRT FP16 | 0.9119 | 0.9524 | 0.8676 | 0.8982 | 0.9245 | 0.5913 |

### Latency / throughput (Jetson Orin, random 192x192 input, preprocessing excluded)

| model | runtime | batch | p50 ms/batch | p95 ms/batch | img/s | mean power mW | mJ/img | peak RAM MB |
|---|---|---|---|---|---|---|---|---|
| mobilenet_v3_large_seed42 | ORT CPU | 1 | 22.3 | 32.0 | 42.2 | 8218 | 195 | 4570 |
| mobilenet_v3_large_seed42 | ORT CPU | 4 | 76.5 | 87.6 | 50.3 | 8229 | 163 | 4591 |
| mobilenet_v3_large_seed42 | ORT CPU | 8 | 150.3 | 195.6 | 50.7 | 8245 | 163 | 4586 |
| mobilenet_v3_large_seed42 | ORT CUDA | 1 | 5.5 | 8.9 | 164.5 | 7996 | 49 | 4859 |
| mobilenet_v3_large_seed42 | ORT CUDA | 4 | 8.5 | 11.0 | 460.5 | 13044 | 28 | 4903 |
| mobilenet_v3_large_seed42 | ORT CUDA | 8 | 15.3 | 15.6 | 520.8 | 15586 | 30 | 5024 |
| mobilenet_v3_large_seed42 | TensorRT FP16 | 1 | 4.0 | 4.2 | 249.8 | 6439 | 26 | 4732 |
| mobilenet_v3_large_seed42 | TensorRT FP16 | 4 | 6.2 | 7.9 | 607.3 | 7410 | 12 | 4732 |
| mobilenet_v3_large_seed42 | TensorRT FP16 | 8 | 7.5 | 8.6 | 1085.8 | 9443 | 9 | 4741 |

## Protocol and caveats

- **Accuracy:** `02_dump_predictions.py`, same eval transform as training (resize 224, centre-crop 192, ImageNet mean/std). The ONNX rows ran on **ONNX Runtime 1.23.2, CPU provider** (the stock aarch64 wheel has no CUDA); the engine rows ran on the Jetson GPU. FP16 engines differ from the ONNX predictions on a few images only (see per-modality columns). MobileNetV3 ONNX accuracy matches the numbers in its own model README.
- **RS** has no `fire`-only images and UAV only 35, so per-class and macro-F1 values there are noisy; the RS macro-F1 includes a zero-score class. Report per modality.
- **Latency:** `benchmark_baseline.py`, 20 warm-up + 100 timed runs per batch size, includes host/device transfer, excludes image decoding/preprocessing. Power is the `VDD_IN` rail from tegrastats during the run; `peak RAM` is whole-system used RAM (it includes about 4.3-4.5 GB of idle usage, so it is not the model footprint).
- **ORT CUDA** rows use the Jetson AI Lab `onnxruntime-gpu 1.23.0` build with the system JetPack cuBLAS/cuDNN preloaded (pip torch's bundled copies otherwise break it). **ORT CPU** rows use the same build with `--onnx-cpu`.
- **SwinV2 engines were built with a batch-1 profile**, so they are benchmarked at batch 1 only. The MobileNetV3 engine was built with max batch 8 and is benchmarked at batch 1/4/8.
- **ORT CPU, SwinV2, batch 1 is unstable** (p50 1022 / 331 / 322 ms for the three models, same architecture); treat it as indicative until rerun with locked clocks. CUDA and TensorRT rows are consistent across models.
- No GPU PyTorch rows: the pip torch build on this board has no sm_87 kernels.
- Environment: Jetson Orin (JetPack 6, L4T R36.4.7), TensorRT 10.3.0, CUDA 12.6.

## Previous contents

The earlier contents of this branch (a superseded 3-class Swin export and its benchmark/figure scripts) were removed and remain in git history.
