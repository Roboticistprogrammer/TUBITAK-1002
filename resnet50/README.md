# resnet50

ResNet-50 (torchvision, ImageNet-1k init, 23.5M params), seed 42. Trained under the same protocol as the SwinV2-B students and MobileNetV3-Large: FASDD CV + RS + UAV train splits, 20 epochs, batch 16, AdamW lr 3e-4 (not tuned), best checkpoint by mean validation accuracy (epoch 20, 0.9064).

## Accuracy on the test dataset (20,435 images)

| model | runtime | CV acc | UAV acc | RS acc | CV macro-F1 | UAV macro-F1 | RS macro-F1 |
|---|---|---|---|---|---|---|---|
| resnet50_seed42 | ONNX Runtime (CPU) | 0.9032 | 0.9467 | 0.8541 | 0.8908 | 0.9015 | 0.5649 |
| resnet50_seed42 | PyTorch (RTX 3070) | 0.9032 | 0.9469 | 0.8541 | 0.8907 | 0.9018 | 0.5649 |

ONNX and PyTorch agree on all RS predictions and on all but one prediction each in CV (1 of 15,884) and UAV (1 of 4,181). TensorRT FP16 and Jetson latency/power have not been measured yet for this model.

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

- `evaluation/resnet50_seed42_eval.json`: per-domain test metrics (accuracy, balanced accuracy, per-class precision/recall/F1, confusion matrices) for the PyTorch and ONNX models, and PyTorch-vs-ONNX agreement.
- `benchmarks/resnet50_seed42_rtx3070_bench.json`: the desktop timing above, with percentiles per batch size.

The weights and ONNX file are in the `amir-salar/models` repository under `baselines/resnet50/seed42/`.

See the top-level README for the protocol and caveats. Macro-F1 follows the convention of the other tables: the mean over all four classes, so the RS value includes the `fire` class that has no test images (score 0); over the classes present, RS macro-F1 is 0.7531.
