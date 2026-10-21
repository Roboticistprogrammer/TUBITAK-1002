# mobilenet_v3_large

## Accuracy on the test dataset (20,435 images)

| model | runtime | CV acc | UAV acc | RS acc | CV macro-F1 | UAV macro-F1 | RS macro-F1 |
|---|---|---|---|---|---|---|---|
| mobilenet_v3_large_seed42 | ONNX Runtime (CPU) | 0.9117 | 0.9522 | 0.8649 | 0.8980 | 0.9243 | 0.5882 |
| mobilenet_v3_large_seed42 | TensorRT FP16 | 0.9119 | 0.9524 | 0.8676 | 0.8982 | 0.9245 | 0.5913 |

## Latency / throughput on Jetson Orin

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

## Figures and full metrics

`figures/` holds the thesis figures (PDF for LaTeX, PNG at 300 dpi) and `metrics.md` / `metrics.json` (balanced accuracy, macro-F1 over present classes, fire / any-fire recall, false-alarm rate, bootstrap 95% CIs, McNemar tests against the SwinV2-B students, ONNX-vs-TensorRT disagreements, filename-prefix label check). Regenerate with `python scripts/plot_baseline_results.py`; pass `--arch-dir resnet50 --model resnet50_seed42 --label ResNet-50` for the next baseline.

See the top-level README for the protocol and caveats.
