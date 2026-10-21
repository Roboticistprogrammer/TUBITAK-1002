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
comparison/            cross-architecture figures and summary table (see comparison/README.md)
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
| kd_routed_seed42 | ORT CPU | 1 | 380.0 | 527.3 | 2.4 | 8727 | 3598 | 4610 |
| kd_routed_seed42 | ORT CPU | 4 | 1314.4 | 1382.9 | 3.1 | 8965 | 2935 | 4636 |
| kd_routed_seed42 | ORT CPU | 8 | 2506.5 | 3253.4 | 3.1 | 9032 | 2924 | 4727 |
| kd_routed_seed42 | ORT CUDA | 1 | 35.0 | 35.5 | 28.5 | 16405 | 576 | 4652 |
| kd_routed_seed42 | ORT CUDA | 4 | 91.2 | 92.8 | 43.6 | 19260 | 441 | 4665 |
| kd_routed_seed42 | ORT CUDA | 8 | 166.6 | 170.5 | 47.8 | 20082 | 420 | 5018 |
| kd_routed_seed42 | TensorRT FP16 | 1 | 11.8 | 17.8 | 79.3 | 13068 | 165 | 4624 |
| kd_uniform_seed42 | ORT CPU | 1 | 362.5 | 427.6 | 2.7 | 8773 | 3270 | 4696 |
| kd_uniform_seed42 | ORT CPU | 4 | 1289.4 | 1352.7 | 3.1 | 8964 | 2880 | 4698 |
| kd_uniform_seed42 | ORT CPU | 8 | 2437.9 | 2564.4 | 3.3 | 8804 | 2689 | 4760 |
| kd_uniform_seed42 | ORT CUDA | 1 | 34.8 | 35.7 | 28.5 | 15946 | 559 | 4704 |
| kd_uniform_seed42 | ORT CUDA | 4 | 91.2 | 91.7 | 43.8 | 18826 | 430 | 5064 |
| kd_uniform_seed42 | ORT CUDA | 8 | 164.9 | 166.3 | 48.5 | 19644 | 405 | 5074 |
| kd_uniform_seed42 | TensorRT FP16 | 1 | 11.9 | 15.6 | 80.7 | 12826 | 159 | 4673 |
| nokd_seed42 | ORT CPU | 1 | 343.4 | 401.7 | 2.8 | 8234 | 2903 | 4642 |
| nokd_seed42 | ORT CPU | 4 | 1229.5 | 1275.0 | 3.3 | 8416 | 2569 | 4640 |
| nokd_seed42 | ORT CPU | 8 | 2377.8 | 2442.9 | 3.4 | 8526 | 2544 | 4690 |
| nokd_seed42 | ORT CUDA | 1 | 34.9 | 35.6 | 28.5 | 15865 | 557 | 4696 |
| nokd_seed42 | ORT CUDA | 4 | 90.9 | 91.3 | 44.0 | 18855 | 429 | 5021 |
| nokd_seed42 | ORT CUDA | 8 | 164.7 | 166.7 | 48.5 | 19639 | 405 | 5027 |
| nokd_seed42 | TensorRT FP16 | 1 | 11.7 | 17.4 | 80.2 | 12522 | 156 | 4614 |

## MobileNetV3-Large baseline (4.2M params)

### Accuracy

| model | runtime | CV acc | UAV acc | RS acc | CV macro-F1 | UAV macro-F1 | RS macro-F1 |
|---|---|---|---|---|---|---|---|
| mobilenet_v3_large_seed42 | ONNX Runtime (CPU) | 0.9117 | 0.9522 | 0.8649 | 0.8980 | 0.9243 | 0.5882 |
| mobilenet_v3_large_seed42 | TensorRT FP16 | 0.9119 | 0.9524 | 0.8676 | 0.8982 | 0.9245 | 0.5913 |

### Latency / throughput (Jetson Orin, random 192x192 input, preprocessing excluded)

| model | runtime | batch | p50 ms/batch | p95 ms/batch | img/s | mean power mW | mJ/img | peak RAM MB |
|---|---|---|---|---|---|---|---|---|
| mobilenet_v3_large_seed42 | ORT CPU | 1 | 22.5 | 30.4 | 42.0 | 7625 | 182 | 4474 |
| mobilenet_v3_large_seed42 | ORT CPU | 4 | 75.9 | 87.0 | 52.0 | 7635 | 147 | 4501 |
| mobilenet_v3_large_seed42 | ORT CPU | 8 | 149.6 | 159.3 | 53.0 | 7674 | 145 | 4533 |
| mobilenet_v3_large_seed42 | ORT CUDA | 1 | 5.7 | 8.8 | 165.5 | 7214 | 44 | 4526 |
| mobilenet_v3_large_seed42 | ORT CUDA | 4 | 8.5 | 11.9 | 434.6 | 11537 | 27 | 4537 |
| mobilenet_v3_large_seed42 | ORT CUDA | 8 | 15.3 | 15.5 | 521.6 | 15262 | 29 | 4609 |
| mobilenet_v3_large_seed42 | TensorRT FP16 | 1 | 4.6 | 4.8 | 219.5 | 5963 | 27 | 4291 |
| mobilenet_v3_large_seed42 | TensorRT FP16 | 4 | 5.3 | 6.4 | 718.2 | 7163 | 10 | 4292 |
| mobilenet_v3_large_seed42 | TensorRT FP16 | 8 | 7.0 | 7.8 | 1143.9 | 8987 | 8 | 4294 |

## ResNet-50 baseline (23.5M params)

### Accuracy

| model | runtime | CV acc | UAV acc | RS acc | CV macro-F1 | UAV macro-F1 | RS macro-F1 |
|---|---|---|---|---|---|---|---|
| resnet50_seed42 | ONNX Runtime (CPU) | 0.9032 | 0.9464 | 0.8541 | 0.8908 | 0.9013 | 0.5649 |
| resnet50_seed42 | TensorRT FP16 | 0.9030 | 0.9467 | 0.8541 | 0.8905 | 0.9015 | 0.5649 |

### Latency / throughput (Jetson Orin, random 192x192 input, preprocessing excluded)

| model | runtime | batch | p50 ms/batch | p95 ms/batch | img/s | mean power mW | mJ/img | peak RAM MB |
|---|---|---|---|---|---|---|---|---|
| resnet50_seed42 | ORT CPU | 1 | 85.5 | 120.8 | 11.3 | 8066 | 711 | 4308 |
| resnet50_seed42 | ORT CPU | 4 | 335.4 | 382.8 | 11.6 | 7936 | 684 | 4343 |
| resnet50_seed42 | ORT CPU | 8 | 662.0 | 715.4 | 11.9 | 8016 | 672 | 4388 |
| resnet50_seed42 | ORT CUDA | 1 | 10.2 | 11.5 | 94.3 | 12503 | 133 | 4512 |
| resnet50_seed42 | ORT CUDA | 4 | 25.7 | 25.8 | 155.8 | 17961 | 115 | 4527 |
| resnet50_seed42 | ORT CUDA | 8 | 43.1 | 43.3 | 185.6 | 18599 | 100 | 4537 |
| resnet50_seed42 | TensorRT FP16 | 1 | 6.5 | 6.7 | 156.6 | 6749 | 43 | 4257 |
| resnet50_seed42 | TensorRT FP16 | 4 | 7.3 | 9.4 | 532.2 | 11158 | 21 | 4266 |
| resnet50_seed42 | TensorRT FP16 | 8 | 10.9 | 11.0 | 737.5 | 16273 | 22 | 4272 |

Figures (PDF + PNG) and full metrics are in `resnet50/figures/`. 

## Protocol and caveats

- **Accuracy:** `02_dump_predictions.py`, same eval transform as training (resize 224, centre-crop 192, ImageNet mean/std). The ONNX rows ran on **ONNX Runtime 1.23.2, CPU provider** (the stock aarch64 wheel has no CUDA); the engine rows ran on the Jetson GPU. FP16 engines differ from the ONNX predictions on a few images only (see per-modality columns). MobileNetV3 ONNX accuracy matches the numbers in its own model README.
- **RS** has no `fire`-only images and UAV only 35, so per-class and macro-F1 values there are noisy; the RS macro-F1 includes a zero-score class. Report per modality.
- **Latency:** `benchmark_baseline.py`, 20 warm-up + 100 timed runs per batch size, includes host/device transfer, excludes image decoding/preprocessing. Power is the `VDD_IN` rail from tegrastats during the run; `peak RAM` is whole-system used RAM (it includes about 4.2-4.4 GB of idle usage, so it is not the model footprint; each JSON records `idle_ram_used_mb`). All 15 benchmark runs were repeated in one session with Firefox closed so RAM and ORT CPU rows are comparable across models; the CPU governor was `schedutil` (clocks could not be locked without sudo) in power mode 25W, see `env.txt`.
- **ORT CUDA** rows use the Jetson AI Lab `onnxruntime-gpu 1.23.0` build with the system JetPack cuBLAS/cuDNN preloaded (pip torch's bundled copies otherwise break it). **ORT CPU** rows use the same build with `--onnx-cpu`.
- **SwinV2 engines were built with a batch-1 profile**, so they are benchmarked at batch 1 only. The MobileNetV3 engine was built with max batch 8 and is benchmarked at batch 1/4/8.
- **Run-to-run variation:** clocks were not locked, so small models at batch 1 vary by roughly 15-25% between sessions (ResNet-50 TensorRT p50 was 5.2 ms in an earlier session and 6.5 ms in the final one; MobileNetV3 4.0 vs 4.6 ms). SwinV2 and the batch-8 rows are stable within a few percent. Compare models within this table, not across sessions.
- No GPU PyTorch rows: the pip torch build on this board has no sm_87 kernels.
- Environment: Jetson Orin (JetPack 6, L4T R36.4.7), TensorRT 10.3.0, CUDA 12.6.

## Previous contents

The earlier contents of this branch (a superseded 3-class Swin export and its benchmark/figure scripts) were removed and remain in git history.
