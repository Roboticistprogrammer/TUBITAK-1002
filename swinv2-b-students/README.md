# swinv2-b-students

## Accuracy on the test dataset (20,435 images)

| model | runtime | CV acc | UAV acc | RS acc | CV macro-F1 | UAV macro-F1 | RS macro-F1 |
|---|---|---|---|---|---|---|---|
| kd_routed_seed42 | ONNX Runtime (CPU) | 0.9182 | 0.9416 | 0.9000 | 0.9029 | 0.8948 | 0.6075 |
| kd_routed_seed42 | TensorRT FP16 | 0.9181 | 0.9416 | 0.9000 | 0.9028 | 0.8948 | 0.6075 |
| kd_uniform_seed42 | ONNX Runtime (CPU) | 0.9168 | 0.9514 | 0.8703 | 0.9047 | 0.9134 | 0.5986 |
| kd_uniform_seed42 | TensorRT FP16 | 0.9169 | 0.9519 | 0.8703 | 0.9047 | 0.9138 | 0.5986 |
| nokd_seed42 | ONNX Runtime (CPU) | 0.9105 | 0.9388 | 0.8838 | 0.8997 | 0.9015 | 0.6163 |
| nokd_seed42 | TensorRT FP16 | 0.9104 | 0.9390 | 0.8865 | 0.8996 | 0.9017 | 0.6177 |

## Latency / throughput on Jetson Orin

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

See the top-level README for the protocol and caveats.
