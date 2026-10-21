# resnet50

ResNet-50 (torchvision, ImageNet-1k init, 23.5M params), seed 42. Trained under the same protocol as the SwinV2-B students and MobileNetV3-Large: FASDD CV + RS + UAV train splits, 20 epochs, batch 16, AdamW lr 3e-4 (not tuned), best checkpoint by mean validation accuracy (epoch 20, 0.9064).

## Accuracy on the test dataset (20,435 images)

| model | runtime | CV acc | UAV acc | RS acc | CV macro-F1 | UAV macro-F1 | RS macro-F1 |
|---|---|---|---|---|---|---|---|
| resnet50_seed42 | ONNX Runtime (CPU) | 0.9032 | 0.9464 | 0.8541 | 0.8908 | 0.9013 | 0.5649 |
| resnet50_seed42 | TensorRT FP16 | 0.9030 | 0.9467 | 0.8541 | 0.8905 | 0.9015 | 0.5649 |

The ONNX Runtime and TensorRT FP16 rows were produced on the Jetson Orin (ONNX Runtime 1.23.2 CPU provider; engine built on the board). ONNX and TensorRT FP16 disagree on 8 CV images, 1 UAV image and none in RS.

## Latency / throughput on Jetson Orin

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

## Files

- `predictions/resnet50_seed42_{onnx,trt_fp16}/`: per-image logits, labels and file names per modality, plus `meta.json`.
- `benchmarks/resnet50_seed42_{ort_cpu,ort_cuda,trt_fp16}.json`: Jetson Orin latency percentiles, power and RAM per batch size.
- `figures/`: thesis figures (PDF + PNG) and `metrics.md` / `metrics.json`; regenerate with `python scripts/plot_baseline_results.py --arch-dir resnet50 --model resnet50_seed42 --label ResNet-50`.

The weights and ONNX file are in the `amir-salar/models` repository under `baselines/resnet50/seed42/`.

See the top-level README for the protocol and caveats. Macro-F1 follows the convention of the other tables: the mean over all four classes, so the RS value includes the `fire` class that has no test images (score 0); over the classes present, RS macro-F1 is 0.7531.
