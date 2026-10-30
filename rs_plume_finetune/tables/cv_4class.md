# 4-class CV: accuracy / macro-F1

| Model | Accuracy % (95% CI) | Macro-F1 % |
|---|---|---|
| kd_routed_seed42_trt_fp16 | 91.8 [91.4, 92.2] | 90.3 |
| kd_routed_finetuned_seed42_trt_fp16 | 93.4 [93.0, 93.7] | 91.8 |
| kd_routed_noplume_seed42_trt_fp16 | 93.2 [92.8, 93.6] | 91.7 |
| kd_uniform_seed42_trt_fp16 | 91.7 [91.3, 92.1] | 90.5 |
| kd_uniform_finetuned_seed42_trt_fp16 | 91.5 [91.1, 92.0] | 90.3 |
| nokd_seed42_trt_fp16 | 91.0 [90.6, 91.5] | 90.0 |
| nokd_finetuned_seed42_trt_fp16 | 91.8 [91.3, 92.2] | 90.7 |
| mobilenet_v3_large_seed42_trt_fp16 | 91.2 [90.7, 91.7] | 89.8 |
| mobilenet_v3_large_finetuned_seed42_trt_fp16 | 90.8 [90.4, 91.3] | 89.5 |
| mobilenet_v3_large_noplume_seed42_trt_fp16 | 91.5 [91.1, 91.9] | 90.1 |
| resnet50_seed42_trt_fp16 | 90.3 [89.9, 90.8] | 89.1 |
| resnet50_finetuned_seed42_trt_fp16 | 88.6 [88.1, 89.0] | 87.4 |

## Exact McNemar on binary correctness (Holm-corrected within this test set)

| A | B | A-only correct | B-only correct | p (Holm) | significant |
|---|---|---|---|---|---|
| kd_routed_seed42_trt_fp16 | kd_routed_finetuned_seed42_trt_fp16 | 971 | 1221 | 6.093e-07 | yes |
| kd_uniform_seed42_trt_fp16 | kd_uniform_finetuned_seed42_trt_fp16 | 1236 | 1211 | 0.6276 | no (inconclusive) |
| nokd_seed42_trt_fp16 | nokd_finetuned_seed42_trt_fp16 | 1198 | 1312 | 0.09633 | no (inconclusive) |
| mobilenet_v3_large_seed42_trt_fp16 | mobilenet_v3_large_finetuned_seed42_trt_fp16 | 1333 | 1276 | 0.5458 | no (inconclusive) |
| resnet50_seed42_trt_fp16 | resnet50_finetuned_seed42_trt_fp16 | 469 | 191 | 4.648e-27 | yes |
| kd_routed_finetuned_seed42_trt_fp16 | kd_routed_noplume_seed42_trt_fp16 | 101 | 76 | 0.2128 | no (inconclusive) |
| mobilenet_v3_large_finetuned_seed42_trt_fp16 | mobilenet_v3_large_noplume_seed42_trt_fp16 | 204 | 312 | 1.141e-05 | yes |
