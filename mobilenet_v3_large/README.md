# mobilenet_v3_large

## Accuracy on the test dataset (20,435 images)

| model | runtime | CV acc | UAV acc | RS acc | CV macro-F1 | UAV macro-F1 | RS macro-F1 |
|---|---|---|---|---|---|---|---|
| mobilenet_v3_large_seed42 | ONNX Runtime (CPU) | 0.9117 | 0.9522 | 0.8649 | 0.8980 | 0.9243 | 0.5882 |
| mobilenet_v3_large_seed42 | TensorRT FP16 | 0.9119 | 0.9524 | 0.8676 | 0.8982 | 0.9245 | 0.5913 |

## Latency / throughput on Jetson Orin

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

## Figures and full metrics

`figures/` holds the thesis figures (PDF for LaTeX, PNG at 300 dpi) and `metrics.md` / `metrics.json` (balanced accuracy, macro-F1 over present classes, fire / any-fire recall, false-alarm rate, bootstrap 95% CIs, McNemar tests against the SwinV2-B students, ONNX-vs-TensorRT disagreements, filename-prefix label check). Regenerate with `python scripts/plot_baseline_results.py`; pass `--arch-dir resnet50 --model resnet50_seed42 --label ResNet-50` for the next baseline.

See the top-level README for the protocol and caveats.
