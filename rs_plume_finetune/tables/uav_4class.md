# 4-class UAV: accuracy / macro-F1

| Model | Accuracy % (95% CI) | Macro-F1 % |
|---|---|---|
| kd_routed_seed42_trt_fp16 | 94.2 [93.4, 94.8] | 89.5 |
| kd_routed_finetuned_seed42_trt_fp16 | 95.1 [94.5, 95.8] | 91.0 |
| kd_routed_noplume_seed42_trt_fp16 | 95.0 [94.3, 95.6] | 90.5 |
| kd_uniform_seed42_trt_fp16 | 95.2 [94.5, 95.8] | 91.4 |
| kd_uniform_finetuned_seed42_trt_fp16 | 94.4 [93.7, 95.0] | 78.7 |
| nokd_seed42_trt_fp16 | 93.9 [93.2, 94.6] | 90.2 |
| nokd_finetuned_seed42_trt_fp16 | 95.3 [94.6, 95.9] | 90.3 |
| mobilenet_v3_large_seed42_trt_fp16 | 95.2 [94.6, 95.9] | 92.4 |
| mobilenet_v3_large_finetuned_seed42_trt_fp16 | 95.1 [94.5, 95.7] | 92.4 |
| mobilenet_v3_large_noplume_seed42_trt_fp16 | 95.4 [94.7, 96.0] | 91.8 |
| resnet50_seed42_trt_fp16 | 94.7 [94.0, 95.4] | 90.2 |
| resnet50_finetuned_seed42_trt_fp16 | 95.0 [94.3, 95.6] | 90.9 |

## Exact McNemar on binary correctness (Holm-corrected within this test set)

| A | B | A-only correct | B-only correct | p (Holm) | significant |
|---|---|---|---|---|---|
| kd_routed_seed42_trt_fp16 | kd_routed_finetuned_seed42_trt_fp16 | 11 | 52 | 1.004e-06 | yes |
| kd_uniform_seed42_trt_fp16 | kd_uniform_finetuned_seed42_trt_fp16 | 56 | 22 | 0.0007474 | yes |
| nokd_seed42_trt_fp16 | nokd_finetuned_seed42_trt_fp16 | 27 | 85 | 2.506e-07 | yes |
| mobilenet_v3_large_seed42_trt_fp16 | mobilenet_v3_large_finetuned_seed42_trt_fp16 | 27 | 22 | 0.6301 | no (inconclusive) |
| resnet50_seed42_trt_fp16 | resnet50_finetuned_seed42_trt_fp16 | 19 | 31 | 0.4757 | no (inconclusive) |
| kd_routed_finetuned_seed42_trt_fp16 | kd_routed_noplume_seed42_trt_fp16 | 15 | 8 | 0.6301 | no (inconclusive) |
| mobilenet_v3_large_finetuned_seed42_trt_fp16 | mobilenet_v3_large_noplume_seed42_trt_fp16 | 30 | 40 | 0.6301 | no (inconclusive) |
