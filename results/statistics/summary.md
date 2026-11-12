# Classification statistics

| group | seeds | CV | UAV | RS | macro | weighted | macro-F1 |
|---|---|---|---|---|---|---|---|
| kd_routed_onnx | 1 | 91.82 | 94.16 | 90.00 | 91.99 | 92.26 | 86.92 |
| kd_routed_trt_fp16 | 3 | 92.70 ± 0.88 | 94.40 ± 0.21 | 88.56 ± 1.36 | 91.88 ± 0.32 | 92.97 ± 0.70 | 86.58 ± 1.08 |
| kd_uniform_onnx | 1 | 91.68 | 95.14 | 87.03 | 91.29 | 92.31 | 87.21 |
| kd_uniform_trt_fp16 | 3 | 91.38 ± 0.97 | 94.79 ± 0.42 | 87.39 ± 0.87 | 91.19 ± 0.51 | 92.01 ± 0.75 | 86.82 ± 0.43 |
| nokd_noamp_trt_fp16 | 3 | 89.71 ± 1.59 | 94.07 ± 0.44 | 88.38 ± 1.18 | 90.72 ± 0.63 | 90.58 ± 1.21 | 86.16 ± 0.90 |
| nokd_onnx | 1 | 91.05 | 93.88 | 88.38 | 91.10 | 91.58 | 87.43 |
| nokd_trt_fp16 | 1 | 91.04 | 93.90 | 88.65 | 91.19 | 91.58 | 87.50 |
| teacher_cv | 1 | 93.44 | 60.99 | 44.59 | 66.34 | 85.92 | 59.17 |
| teacher_ensemble | 1 | 93.44 | 94.95 | 92.16 | 93.52 | 93.73 | 88.88 |
| teacher_rs | 1 | 54.39 | 69.48 | 92.16 | 72.01 | 58.16 | 53.95 |
| teacher_uav | 1 | 52.51 | 94.95 | 38.11 | 61.86 | 60.93 | 56.51 |

## McNemar (exact, two-sided)

| comparison | seed | domain | only A correct | only B correct | p | p (Holm) |
|---|---|---|---|---|---|---|
| kd_routed vs nokd_noamp | 42 | cv | 725 | 517 | 3.933e-09 | 5.899e-08 |
| kd_routed vs nokd_noamp | 42 | uav | 79 | 54 | 0.03703 | 0.4443 |
| kd_routed vs nokd_noamp | 42 | rs | 22 | 11 | 0.08014 | 0.8675 |
| kd_routed vs nokd_noamp | 43 | cv | 1122 | 216 | 8.945e-148 | 1.61e-146 |
| kd_routed vs nokd_noamp | 43 | uav | 78 | 69 | 0.5095 | 1 |
| kd_routed vs nokd_noamp | 43 | rs | 17 | 19 | 0.8679 | 1 |
| kd_routed vs nokd_noamp | 44 | cv | 775 | 464 | 8.492e-19 | 1.359e-17 |
| kd_routed vs nokd_noamp | 44 | uav | 77 | 70 | 0.6208 | 1 |
| kd_routed vs nokd_noamp | 44 | rs | 15 | 22 | 0.324 | 1 |
| kd_uniform vs nokd_noamp | 42 | cv | 912 | 723 | 3.25e-06 | 4.225e-05 |
| kd_uniform vs nokd_noamp | 42 | uav | 109 | 41 | 2.614e-08 | 3.659e-07 |
| kd_uniform vs nokd_noamp | 42 | rs | 20 | 20 | 1 | 1 |
| kd_uniform vs nokd_noamp | 43 | cv | 1321 | 639 | 1.763e-54 | 2.997e-53 |
| kd_uniform vs nokd_noamp | 43 | uav | 76 | 71 | 0.7416 | 1 |
| kd_uniform vs nokd_noamp | 43 | rs | 17 | 19 | 0.8679 | 1 |
| kd_uniform vs nokd_noamp | 44 | cv | 803 | 876 | 0.07886 | 0.8675 |
| kd_uniform vs nokd_noamp | 44 | uav | 79 | 61 | 0.1505 | 1 |
| kd_uniform vs nokd_noamp | 44 | rs | 18 | 27 | 0.2327 | 1 |
