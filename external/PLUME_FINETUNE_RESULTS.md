# Plume fine-tuning: validation results (seed 42)

Branch `retrain/rs-smoke-plumes`; protocol: `external/PLUME_PROTOCOL.md` (Sections 7 and 8). Runs were made on the Windows
machine (RTX 3070, 8 GB) between 2026-10-23 and 2026-10-25 with `scripts/plumes/run_finetuning.py`. The driver finished with
`ALL RUNS FINISHED` (`outputs/plume_finetune/driver.log`).

**These are validation numbers, not test results.** The same validation sets (CV, UAV, FASDD-RS val and plume val) pick the
checkpoint, so every "after" value is somewhat optimistic. The test evaluation of the protocol (TensorRT FP16 on the Jetson,
bootstrap confidence intervals, McNemar with Holm correction, plume test groups, CV/UAV forgetting check) has **not** been run
yet. Single seed (42) for every model.

## What was run

Five epochs at one tenth of each model's original learning rate, with the plume images added (treatment) or not (control,
`--no-plumes`). Control runs exist for the RS teacher, the routed student and MobileNetV3-Large.

| Run | Best epoch | Selection score before -> after | FASDD-RS val BA drop (pts) | `both` recall drop (pts) | Fallback triggered | Time (min) |
|---|---|---|---|---|---|---|
| RS teacher | 4 | 0.759 -> 0.915 | -0.62 | 1.57 | no | 27 |
| RS teacher, control (no plumes) | 3 | 0.759 -> 0.769 | -0.34 | 1.05 | no | 15 |
| KD routed | 5 | 0.878 -> 0.934 | -0.34 | 0.00 | no | 359 |
| KD routed, control | 5 | 0.878 -> 0.885 | -0.45 | -0.26 | no | 335 |
| KD uniform | 5 | 0.871 -> 0.913 | -0.96 | -0.26 | no | 359 |
| No KD | 4 | 0.869 -> 0.928 | -0.73 | -1.05 | no | 280 |
| MobileNetV3-Large | 5 | 0.866 -> 0.917 | -0.22 | -1.31 | no | 205 |
| MobileNetV3-Large, control | 5 | 0.866 -> 0.868 | 0.00 | 0.00 | no | 157 |
| ResNet-50 | 4 | 0.858 -> 0.915 | -0.40 | 1.31 | no | 183 |

A negative drop is an improvement. Neither pre-registered fallback (source-balanced teacher rerun, class-balanced sampler)
was triggered: the largest `both`-recall drop is 1.57 points (RS teacher), under the 2-point limit; no balanced-accuracy drop
on FASDD-RS val is positive for any treatment run. The teacher check passed, so the KD students were fine-tuned from the normal
fine-tuned RS teacher (`driver_decisions.json`).

## Treatment vs control (validation, percent)

| | Plume val BA before | Plume val BA after | Plume val BA, control | FASDD-RS val BA before | FASDD-RS val BA after | FASDD-RS val BA, control |
|---|---|---|---|---|---|---|
| RS teacher | 60.2 | 87.7 | 61.0 | 94.8 | 95.4 | 95.2 |
| KD routed | 55.2 | 84.6 | 56.0 | 94.8 | 95.2 | 95.3 |
| MobileNetV3-Large | 51.4 | 81.3 | 51.3 | 92.1 | 92.4 | 92.1 |

Other treatment-only runs, plume val BA before -> after: KD uniform 61.6 -> 84.0, No KD 54.3 -> 84.1, ResNet-50 50.7 -> 84.0.

CV and UAV 4-class validation accuracy (forgetting check on validation data): KD routed 94.6 -> 95.3 (CV), 94.3 -> 95.1 (UAV);
the control gives 95.3 and 95.2. MobileNetV3-Large 93.6 -> 93.5 (CV), 95.1 -> 95.0 (UAV). No KD 93.5 -> 94.4 and 94.1 -> 95.3.
KD uniform 90.4 -> 90.6 and 94.8 -> 94.2. ResNet-50 92.9 -> 92.4 and 94.5 -> 94.8.

## Observations

- **The gain on plume validation images comes from the plume data.** Plume val balanced accuracy rises by 22 to 33 points
  in every treatment run (smallest for KD uniform, whose before model was already at 61.6), while the matching control runs stay where the before model was (51 to 61%). The control runs also
  change FASDD-RS val by at most 0.5 points.
- **No harm to FASDD-RS validation.** Balanced accuracy on FASDD-RS val is within 1 point of the before model for every model
  (it rises slightly for most). The `both` recall changes by at most 1.6 points.
- **Plume val false-alarm rate falls** in every treatment run (RS teacher 36.9 -> 4.0, KD uniform 35.1 -> 8.9, KD routed 12.2 -> 6.7).
- **The student with the control teacher** (`kd_routed_noplume`) gains nothing on plume val (BA 56.0), so the routed student's
  plume gain is not a by-product of additional epochs or of the changed schedule.
- The plume val gain says the models learned this plume dataset; it does not show that wildfire-smoke detection in FASDD-RS
  improves. That is what the FASDD-RS test (non-inferiority) and the test-set evaluation will answer.

## Caveats

- Validation only; the plume val set is part of the checkpoint selection rule (Section 7 of the protocol).
- One seed. Differences of a few tenths of a point on FASDD-RS val (about 740 images) are within noise.
- `compare_runs.py` prints a different "before" column for each run that has its own epoch-0 evaluation (the KD students, the CNNs
  and the no-KD student start from different checkpoints), so use each run's own before value, as in the tables above.
- Per-epoch logs (`run.log`, 3 to 9 MB each), checkpoints (`best.pt`, `last.pt`) and the smoke-test run are not committed; they
  stay in the main checkout's `outputs/plume_finetune/` on the Windows machine.

## Files

`outputs/plume_finetune/<run>/seed42/{history.json,summary.json}` for the nine runs; `driver.log` and `driver_decisions.json`
(order, timings, teacher check, teachers used); `before_binary_rs/` (binary FASDD-RS test metrics of the before models, as in
Section 1 of the protocol). Reproduce the tables with
`python -I scripts/plumes/compare_runs.py outputs/plume_finetune/<treatment>/seed42 outputs/plume_finetune/<control>/seed42`.

## Not done yet

1. Fine-tuned checkpoints through `02_dump_predictions.py` (`--domains cv rs uav plume`) and the TensorRT FP16 engines on the Jetson.
2. Test-set tables with CIs and McNemar (protocol Section 8).
3. Seeds 43 and 44 for the routed student and MobileNetV3-Large (the seed-44 student folders on the Windows machine hold only a
   placeholder `DONE` file; MobileNetV3-Large seeds 43 and 44 are not trained).
