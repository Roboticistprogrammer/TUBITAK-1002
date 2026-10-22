# External validation on USTC_SmokeRS: evaluation protocol

This protocol is fixed **before** any model is run on USTC_SmokeRS. The commit that adds this file is the reference
point: any change made after the first prediction is run must be recorded under "Deviations" at the end, with the date
and the reason. Nothing in this file is chosen by looking at results.

## 1. Purpose

The FASDD remote-sensing (RS) test split has 370 images and no `fire`-only image, so RS comparisons between models are
underpowered (only accuracy differences above about 3 percentage points can reach p < 0.05 with the exact McNemar test).
This evaluation tests the deployed models, **without retraining or tuning**, on an independent satellite dataset to
measure how they generalize to RS imagery from another sensor and region.

The FASDD test split is not changed. USTC_SmokeRS results are reported separately (thesis Section 4.7).

## 2. Dataset

| Item | Value |
|---|---|
| Name | USTC_SmokeRS |
| Source paper | Ba, R., et al. (2019). SmokeNet: Satellite smoke scene detection using convolutional neural network with spatial and channel-wise attention. *Remote Sensing*, 11(14), 1702. https://doi.org/10.3390/rs11141702 |
| Sensor | MODIS (Terra and Aqua), true-colour RGB composite of bands 1, 4, 3 |
| Images (per paper) | 6,225, 256 × 256 px, `.tif`, about 1 km per pixel |
| Labels | one scene-level label per image: `smoke`, `cloud`, `dust`, `haze`, `land`, `seaside` |
| Licence | none stated; cite the paper; images are not committed to this repository |
| Local path | `datasets/USTC_SmokeRS/` (ignored by `.gitignore` on every branch via `datasets/` and `*.tif`) |

### Download record (fill in when downloaded)

| Item | Value |
|---|---|
| Download source (Baidu Pan link from the paper, lab page, or mirror) | |
| Download date | |
| Archive file name | |
| Archive SHA-256 | |
| Images found per class (smoke / cloud / dust / haze / land / seaside) | |
| Total images found (expected 6,225) | |

A mirror is used only if its per-class counts match the paper.

### Image format check (fill in before running any model)

| Check | Result |
|---|---|
| Bit depth (8-bit or 16-bit) | |
| Channels and order (RGB expected) | |
| Opens correctly with `PIL.Image.open(path).convert("RGB")`, as in `src/firecls/data/dataset.py` | |
| Visual comparison of 10 USTC and 10 FASDD-RS images loaded the same way (brightness, colour) | |

If the images are 16-bit or the channel order differs from FASDD-RS, the loader is fixed **before** any prediction is run,
and the fix is described here.

## 3. Models

All deployable models are evaluated as the artifacts reported in Chapter 4, so the results describe the deployed models.

| Model | Artifact | Runtime |
|---|---|---|
| SwinV2-B student, KD domain-routed | `kd_routed_seed42` FP16 engine | TensorRT FP16, Jetson |
| SwinV2-B student, KD uniform | `kd_uniform_seed42` FP16 engine | TensorRT FP16, Jetson |
| SwinV2-B student, hard labels (no KD) | `nokd_seed42` FP16 engine | TensorRT FP16, Jetson |
| MobileNetV3-Large | `mobilenet_v3_large_seed42` FP16 engine | TensorRT FP16, Jetson |
| ResNet-50 | `resnet50_seed42` FP16 engine | TensorRT FP16, Jetson |
| RS teacher (reference only) | `outputs/teachers/FASDD_RS/best.pt` | PyTorch FP32 |

Seeds 43 and 44 are added with the same protocol once their engines exist.

## 4. Preprocessing

Identical to the FASDD evaluation (`build_eval_transform` in `src/firecls/transforms.py`): resize the shorter side to
224 px, centre-crop 192 × 192, ImageNet mean and standard deviation. For a 256 × 256 USTC image this keeps the central
192 × 192 px of a 224 × 224 resize; FASDD-RS images go through the same transform.

## 5. Label mapping and prediction rule

| USTC label | Binary ground truth |
|---|---|
| `smoke` | positive |
| `cloud`, `dust`, `haze`, `land`, `seaside` | negative |

- **Prediction rule:** an image is predicted positive if the predicted class (argmax of the four logits) is `fire`,
  `smoke` or `both`; it is predicted negative if the predicted class is `neither`.
- **Threshold-free score:** `1 − P(neither)` from the softmax of the logits, used only for ROC-AUC.
- No threshold, temperature or calibration is tuned on USTC_SmokeRS.

## 6. Metrics

For every model:

1. Smoke recall: share of `smoke` images predicted positive.
2. False-alarm rate: share of negative images predicted positive, overall and separately for `cloud`, `dust`, `haze`,
   `land` and `seaside`.
3. Balanced accuracy: mean of smoke recall and (1 − overall false-alarm rate).
4. ROC-AUC of the threshold-free score.
5. Distribution of the four predicted classes on `smoke` images (share predicted `smoke`, `both`, `fire`, `neither`).

## 7. Statistics

- 95% confidence intervals: bootstrap over images, 2,000 resamples, as for the FASDD results.
- Pairwise comparisons: exact two-sided McNemar test on binary correctness, for every pair of the five deployed models
  (10 pairs), with Holm correction across the 10 tests. Report b, c, the difference in balanced accuracy and both the
  unadjusted and Holm-adjusted p-values.
- A non-significant difference is reported as inconclusive, not as evidence of equal performance.

## 8. Reporting rules

- All five deployed models are reported, whatever their results.
- No retraining, fine-tuning or threshold selection on USTC_SmokeRS.
- Results go to the `results` branch under `external/ustc_smokers/`; each `meta.json` records the commit of this branch
  that produced it.
- Only a few example images may appear in the thesis, with a citation; dataset images are never committed.

## 9. Known limitations (to be stated in the thesis)

- USTC_SmokeRS has no fire class, so this evaluation covers smoke detection and false alarms on smoke-like scenes
  (cloud, dust, haze), not fire detection in satellite imagery.
- About 1 km per pixel, a different sensor and a different region from FASDD-RS: the result measures generalization under
  a combined shift in resolution, sensor and geography, which cannot be separated.
- The binary mapping collapses the models' four classes; a model may detect smoke but label it `both`, which still counts
  as a correct positive.

## 10. Deviations

Record any change made after the first prediction was run.

| Date | Change | Reason |
|---|---|---|
| | | |
