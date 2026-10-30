# Binary smoke/fire vs neither: plume

| Model | Recall % (95% CI) | False-alarm % (95% CI) | Balanced acc. % (95% CI) | F1 % | ROC-AUC | Errors |
|---|---|---|---|---|---|---|
| teacher_rs_before | 58.0 [53.2, 62.6] | 34.4 [30.3, 38.6] | 61.8 [58.7, 64.9] | 60.3 | 0.6383 | 344/900 |
| teacher_rs_after | 73.3 [69.1, 77.4] | 4.7 [2.7, 6.8] | 84.3 [82.1, 86.6] | 82.4 | 0.9152 | 141/900 |
| teacher_rs_noplume_after | 46.4 [41.9, 51.2] | 28.7 [24.6, 32.7] | 58.9 [55.8, 62.1] | 53.0 | 0.6164 | 370/900 |
| kd_routed_seed42_trt_fp16 | 23.3 [19.6, 27.2] | 14.7 [11.6, 18.0] | 54.3 [51.8, 56.8] | 33.8 | 0.5544 | 411/900 |
| kd_routed_finetuned_seed42_trt_fp16 | 68.4 [64.1, 72.7] | 6.2 [4.0, 8.7] | 81.1 [78.7, 83.5] | 78.4 | 0.8729 | 170/900 |
| kd_routed_noplume_seed42_trt_fp16 | 24.9 [20.9, 28.8] | 17.3 [13.9, 21.0] | 53.8 [51.2, 56.4] | 35.0 | 0.5545 | 416/900 |
| kd_uniform_seed42_trt_fp16 | 54.4 [50.0, 59.2] | 39.8 [35.2, 44.4] | 57.3 [54.2, 60.7] | 56.1 | 0.6121 | 384/900 |
| kd_uniform_finetuned_seed42_trt_fp16 | 71.6 [67.2, 75.7] | 10.0 [7.4, 12.9] | 80.8 [78.2, 83.2] | 78.8 | 0.8685 | 173/900 |
| nokd_seed42_trt_fp16 | 22.7 [19.1, 26.7] | 17.6 [14.2, 21.0] | 52.6 [50.0, 55.1] | 32.3 | 0.5702 | 427/900 |
| nokd_finetuned_seed42_trt_fp16 | 72.2 [68.0, 76.3] | 6.9 [4.7, 9.3] | 82.7 [80.2, 85.0] | 80.6 | 0.8829 | 156/900 |
| mobilenet_v3_large_seed42_trt_fp16 | 17.1 [13.8, 20.8] | 18.4 [14.8, 22.0] | 49.3 [47.0, 51.9] | 25.2 | 0.5286 | 456/900 |
| mobilenet_v3_large_finetuned_seed42_trt_fp16 | 75.6 [71.6, 79.6] | 16.7 [13.5, 20.3] | 79.4 [76.7, 82.0] | 78.6 | 0.8747 | 185/900 |
| mobilenet_v3_large_noplume_seed42_trt_fp16 | 17.3 [14.0, 21.0] | 16.7 [13.3, 20.1] | 50.3 [47.9, 52.8] | 25.9 | 0.5297 | 447/900 |
| resnet50_seed42_trt_fp16 | 15.3 [12.2, 18.6] | 14.0 [10.8, 17.4] | 50.7 [48.5, 53.0] | 23.7 | 0.5621 | 444/900 |
| resnet50_finetuned_seed42_trt_fp16 | 71.1 [67.0, 75.2] | 11.6 [8.7, 14.8] | 79.8 [77.1, 82.3] | 77.9 | 0.8784 | 182/900 |

## Exact McNemar on binary correctness (Holm-corrected within this test set)

| A | B | A-only correct | B-only correct | BA diff (A-B) pp | p | p (Holm) | significant |
|---|---|---|---|---|---|---|---|
| teacher_rs_before | teacher_rs_after | 58 | 261 | -22.6 | 6.757e-32 | 6.757e-32 | yes |
| kd_routed_seed42_trt_fp16 | kd_routed_finetuned_seed42_trt_fp16 | 38 | 279 | -26.8 | 1.802e-46 | 1.081e-45 | yes |
| kd_uniform_seed42_trt_fp16 | kd_uniform_finetuned_seed42_trt_fp16 | 50 | 261 | -23.4 | 1.311e-35 | 2.622e-35 | yes |
| nokd_seed42_trt_fp16 | nokd_finetuned_seed42_trt_fp16 | 32 | 303 | -30.1 | 1.664e-56 | 1.498e-55 | yes |
| mobilenet_v3_large_seed42_trt_fp16 | mobilenet_v3_large_finetuned_seed42_trt_fp16 | 65 | 336 | -30.1 | 3.841e-45 | 1.92e-44 | yes |
| resnet50_seed42_trt_fp16 | resnet50_finetuned_seed42_trt_fp16 | 49 | 311 | -29.1 | 9.811e-48 | 7.849e-47 | yes |
| teacher_rs_after | teacher_rs_noplume_after | 274 | 45 | +25.4 | 3.366e-41 | 1.01e-40 | yes |
| kd_routed_finetuned_seed42_trt_fp16 | kd_routed_noplume_seed42_trt_fp16 | 285 | 39 | +27.3 | 2.496e-47 | 1.747e-46 | yes |
| mobilenet_v3_large_finetuned_seed42_trt_fp16 | mobilenet_v3_large_noplume_seed42_trt_fp16 | 332 | 70 | +29.1 | 6.799e-42 | 2.72e-41 | yes |
