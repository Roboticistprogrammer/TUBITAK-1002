# resnet50

ResNet-50 (torchvision, ImageNet-1k init, 23.5M params), seed 42. Trained under the same protocol as the SwinV2-B students and MobileNetV3-Large: FASDD CV + RS + UAV train splits, 20 epochs, batch 16, AdamW lr 3e-4 (not tuned), best checkpoint by mean validation accuracy (epoch 20, 0.9064).

## Accuracy on the test dataset (20,435 images)

| model | runtime | CV acc | UAV acc | RS acc | CV macro-F1 | UAV macro-F1 | RS macro-F1 |
|---|---|---|---|---|---|---|---|
| resnet50_seed42 | ONNX Runtime (CPU) | 0.9032 | 0.9464 | 0.8541 | 0.8908 | 0.9013 | 0.5649 |
| resnet50_seed42 | TensorRT FP16 | 0.9030 | 0.9467 | 0.8541 | 0.8905 | 0.9015 | 0.5649 |
| resnet50_seed42 | PyTorch (RTX 3070) | 0.9032 | 0.9469 | 0.8541 | 0.8907 | 0.9018 | 0.5649 |

The ONNX Runtime and TensorRT FP16 rows were produced on the Jetson Orin (ONNX Runtime 1.23.2 CPU provider; engine built on the board). The PyTorch row is from the desktop evaluation (`evaluation/resnet50_seed42_eval.json`; its ONNX-on-desktop-CPU row has UAV 0.9467, one image different from the Jetson run). ONNX and TensorRT FP16 disagree on 8 CV images, 1 UAV image and none in RS.

## Latency / throughput on Jetson Orin

| model | runtime | batch | p50 ms/batch | p95 ms/batch | img/s | mean power mW | mJ/img | peak RAM MB |
|---|---|---|---|---|---|---|---|---|
| resnet50_seed42 | ORT CPU | 1 | 79.6 | 116.7 | 12.0 | 8124 | 678 | 5565 |
| resnet50_seed42 | ORT CPU | 4 | 331.4 | 389.0 | 11.7 | 8002 | 684 | 5591 |
| resnet50_seed42 | ORT CPU | 8 | 665.7 | 769.7 | 11.6 | 8038 | 691 | 5710 |
| resnet50_seed42 | ORT CUDA | 1 | 10.6 | 11.9 | 92.2 | 12533 | 136 | 5993 |
| resnet50_seed42 | ORT CUDA | 4 | 25.7 | 26.1 | 155.1 | 18028 | 116 | 6121 |
| resnet50_seed42 | ORT CUDA | 8 | 43.2 | 43.5 | 185.0 | 18636 | 101 | 6142 |
| resnet50_seed42 | TensorRT FP16 | 1 | 5.2 | 7.0 | 188.6 | 7048 | 37 | 5824 |
| resnet50_seed42 | TensorRT FP16 | 4 | 7.3 | 9.4 | 535.3 | 11200 | 21 | 5826 |
| resnet50_seed42 | TensorRT FP16 | 8 | 10.9 | 11.3 | 731.3 | 16340 | 22 | 5808 |

## Reference speed on a desktop GPU (not Jetson)

Measured on an NVIDIA RTX 3070 workstation (PyTorch 2.14.1+cu130; ONNX Runtime 1.23.2 with the CPU provider only), 20 warm-up + 200 timed runs per batch size, preprocessing excluded.

| model | runtime | batch | p50 ms/batch | p95 ms/batch | img/s |
|---|---|---|---|---|---|
| resnet50_seed42 | PyTorch GPU | 1 | 8.6 | 12.1 | 107.9 |
| resnet50_seed42 | PyTorch GPU | 4 | 8.6 | 12.7 | 436.4 |
| resnet50_seed42 | PyTorch GPU | 8 | 9.3 | 11.0 | 841.6 |
| resnet50_seed42 | ONNX Runtime CPU | 1 | 15.9 | 16.9 | 62.5 |
| resnet50_seed42 | ONNX Runtime CPU | 4 | 60.2 | 68.4 | 65.2 |
| resnet50_seed42 | ONNX Runtime CPU | 8 | 128.6 | 142.1 | 61.3 |

## Files

- `predictions/resnet50_seed42_{onnx,trt_fp16}/`: per-image logits, labels and file names per modality, plus `meta.json`.
- `benchmarks/resnet50_seed42_{ort_cpu,ort_cuda,trt_fp16}.json`: Jetson Orin latency percentiles, power and RAM per batch size.
- `figures/`: thesis figures (PDF + PNG) and `metrics.md` / `metrics.json`; regenerate with `python scripts/plot_baseline_results.py --arch-dir resnet50 --model resnet50_seed42 --label ResNet-50`.
- `evaluation/resnet50_seed42_eval.json`: per-domain test metrics (accuracy, balanced accuracy, per-class precision/recall/F1, confusion matrices) for the PyTorch and ONNX models, and PyTorch-vs-ONNX agreement.
- `benchmarks/resnet50_seed42_rtx3070_bench.json`: the desktop timing above, with percentiles per batch size.

The weights and ONNX file are in the `amir-salar/models` repository under `baselines/resnet50/seed42/`.

See the top-level README for the protocol and caveats. Macro-F1 follows the convention of the other tables: the mean over all four classes, so the RS value includes the `fire` class that has no test images (score 0); over the classes present, RS macro-F1 is 0.7531.
