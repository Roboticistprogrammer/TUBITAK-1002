# Plume fine-tuning: test-set evaluation

Test-set evaluation of the plume fine-tuning campaign (`external/PLUME_PROTOCOL.md`, Section 8; branch
`retrain/rs-smoke-plumes` at `539f2f9`). Seed 42. Students/CNNs: TensorRT FP16 on the Jetson. RS teacher: PyTorch FP32
on the main machine. See `meta.json` for machines, commit, and deviations from the protocol.

## Gate check

The before models' FASDD-RS binary results were recomputed and matched `PLUME_PROTOCOL.md` Section 1's reference
table exactly, for all 6 before models (errors/370: teacher 14, KD routed 17, KD uniform 27, no-KD 26,
MobileNetV3-Large 26, ResNet-50 30).

## Headline results

- **FASDD-RS test (non-inferiority check):** no before-vs-after difference reached significance for any model after
  Holm correction (`tables/rs_mcnemar.csv`) — fine-tuning does no measurable harm on FASDD-RS.
- **Plume test (gain from the added data):** every model improved significantly after fine-tuning (p < 0.0001,
  Holm-corrected), and every treatment model significantly beat its no-plume control (teacher, KD routed,
  MobileNetV3-Large; p < 0.0001). Balanced accuracy rose from ~50-62% (near chance) before/control to ~79-84% after.
- **FASDD CV/UAV test (forgetting check):** mixed, model-dependent:
  - KD routed improved significantly on both CV and UAV after fine-tuning.
  - ResNet-50 dropped significantly on CV after fine-tuning (90.3% -> 88.6% accuracy, p < 1e-26); a small,
    roughly uniform recall decline across all 4 classes rather than one class collapsing.
  - KD uniform's UAV macro-F1 dropped sharply (91.4% -> 78.7%) despite stable overall accuracy (95.2% -> 94.4%):
    recall on the rare `fire`-only UAV class (35 of 4181 images) collapsed from 82.9% to 22.9% while the other three
    classes stayed stable. Worth flagging for the thesis as a localized forgetting effect invisible in accuracy alone.
  - MobileNetV3-Large and No-KD showed no significant CV/UAV change (No-KD) or a significant *improvement*
    (No-KD on UAV); see `tables/cv_4class.md` and `tables/uav_4class.md` for the full picture.

## Contents

- `predictions/<name>/{cv,rs,uav,plume}.npz + meta.json` — per-image logits/labels/names for every before/after/control
  model, plus commit hash and checkpoint/engine SHA-256 in each `meta.json`.
- `tables/rs_binary.{csv,md}`, `tables/plume_binary.{csv,md}` — binary recall/false-alarm/balanced accuracy/F1/ROC-AUC
  with 95% bootstrap CIs (2000 resamples), plus RS recall on the 31 `smoke` and 191 `both` images.
- `tables/rs_mcnemar.csv`, `tables/plume_mcnemar.csv`, `tables/cv_mcnemar.csv`, `tables/uav_mcnemar.csv` — exact
  McNemar on binary correctness, Holm-corrected within each test set.
- `tables/rs_confusion_before_after.{csv,md}` — 2x2 RS confusion matrices, before vs after, per model.
- `tables/cv_4class.{csv,md}`, `tables/uav_4class.{csv,md}` — 4-class accuracy and macro-F1 with 95% CIs.

Latency was not re-benchmarked (architectures unchanged, per protocol).
