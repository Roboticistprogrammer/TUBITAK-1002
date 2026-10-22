# External validation on USTC_SmokeRS: evaluation protocol

This protocol is fixed **before** any model is run on USTC_SmokeRS. The commit that adds this file is the reference
point: any change made after the first prediction is run must be recorded under "Deviations" at the end, with the date
and the reason. Nothing in this file is chosen by looking at results.

Revision note (2026-10-22, before any model was run on USTC_SmokeRS): this revision fills in the download record and the
image format check, and adds the secondary prediction rule and the secondary preprocessing (sensitivity analyses). The
only model runs made so far were on FASDD-RS images and on a 60-image stand-in folder built from FASDD-RS images, to test
the code; none used USTC_SmokeRS images.

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
| Licence | Hugging Face card: "other", "for research/education purposes"; cite the paper; images are not committed to this repository |
| Local path | `datasets/USTC_SmokeRS/<class>/<file>.tif` (ignored by `.gitignore` on every branch via `datasets/` and `*.tif`) |
| Split | none: **all 6,225 images are the test set** (`split=test` for every image). Nothing is trained or tuned on USTC_SmokeRS, so a split would only discard test images. The paper's own random 64/16/20 split is not used. |

### Download record

| Item | Value |
|---|---|
| Download source | Hugging Face dataset `jonathan-roberts1/USTC_SmokeRS`, **revision (commit) `c5d7b1bd3da912bb0b3c1ab5c5e619b23103dc32`** (last modified 2023-03-31), https://huggingface.co/datasets/jonathan-roberts1/USTC_SmokeRS. A mirror can change, so this revision is pinned. Not the Baidu Pan link of the paper (see below). |
| Download date | 2026-10-22 |
| Archive file names and SHA-256 | `data/train-00000-of-00003-2f535e0689383c43.parquet` `fd4fa280e5c981dc332c0b91075513a0165eff6e84618ffa62d5c90edbe36d33`; `data/train-00001-of-00003-3dab9bc268d0c492.parquet` `ba7061c4c52382b4cdf66769c85492d2124b73a8e591a5d4808f43e64b9ea9e6`; `data/train-00002-of-00003-664bfe08807e3230.parquet` `64f819da123ac39887cfd62b4ba3c45543597367294283645a4fe9e7d1c485ed` (2,075 rows each) |
| Unpacked with | `scripts/external/unpack_ustc_parquet.py` (needs `pyarrow`, run once in a throwaway environment; `pyarrow` is not a project dependency). It writes `datasets/USTC_SmokeRS/manifest.csv` (file, class, hf_label, sha256), SHA-256 `0df57d22f1505b36a72716ee8274ac7ad0726bfe02aeb3e05a437d4a10bbd240`. |
| Images found per class (cloud / dust / haze / land / seaside / smoke) | 1,164 / 1,009 / 1,002 / 1,027 / 1,007 / 1,016 |
| Total images found (expected 6,225) | 6,225 |

**Per-class counts against the paper.** Table 2 of Ba et al. (2019), https://www.mdpi.com/2072-4292/11/14/1702 (read
from the publisher's open-access PDF on 2026-10-22): Cloud 1164, Dust 1009, Haze 1002, Land 1027, Seaside 1007,
Smoke 1016, total 6,225. All six counts match the downloaded data exactly, so the mirror is accepted under the rule
that a mirror is used only if its per-class counts match the paper.

**What the copy is, and is not.** The parquet stores each image as its encoded TIFF (`image.bytes`, all 6,225 start with
the TIFF header) with the original file name (`image.path`, e.g. `cloud_1.tif`). The script writes these bytes to disk
unchanged (no decoding or re-encoding) under the original names, and checked that the SHA-256 of all 6,225 files read
back from disk equals that of the parquet bytes. The original Baidu Pan archive was not obtained, so byte-identity with
it cannot be proven; what agrees with the paper is the count per class, the `.tif` format, the 256 × 256 size and the
file naming by class. Every file name has the form `<class>_<number>.tif` and all names and image bytes are unique (no
duplicate image, none under two labels).

**Label order.** The Hugging Face integer `label` uses the order 0 cloud, 1 dust, 2 haze, 3 land, 4 seaside, 5 smoke,
which differs from the order used in this protocol (smoke first). The class of every image is therefore taken from the
class name in its file name, never from the integer; the unpack script checks that name and integer agree for all 6,225
images and the index (`data_index/ustc_smokers.csv`) stores the class name (`label6`) and the binary label (`label_bin`).

Earlier download attempts (2026-10-22, before the Hugging Face copy was used): the lab page
https://complex.ustc.edu.cn/sjwwataset/list.htm lists a OneDrive link (anonymous access refused: HTTP 403 on the page,
`unauthenticated` from the share API) and `http://rec.ustc.edu.cn/share/ba4ac700-b991-11e9-b7ad-f1507c2962da` (its share
API answered "login timeout" and "share number does not exist"). The Baidu Pan link in the paper was not tried.

### Image format check

Done on all 6,225 unpacked files before any model was run.

| Check | Result |
|---|---|
| Bit depth (8-bit or 16-bit) | 8-bit: TIFF `BitsPerSample` = (8, 8, 8) and NumPy dtype `uint8` for all 6,225 images; pixel values use the full 0 to 255 range |
| Channels and order (RGB expected) | 3 channels, PIL mode `RGB` for all 6,225 images; 256 × 256 px for all. The files are the MODIS band 1, 4, 3 composite stored as RGB; the contact sheet shows natural colours (vegetation green, sea blue, desert orange), so the channel order is not swapped |
| Opens correctly with `PIL.Image.open(path).convert("RGB")`, as in `src/firecls/data/dataset.py` | Yes: 6,225 of 6,225 opened and converted without error |
| Visual comparison of 10 USTC and 10 FASDD-RS images loaded the same way (brightness, colour) | Both loaded with `PIL.Image.open(...).convert("RGB")` and shown after resize 224 and centre crop 192 (random seed 0; 10 USTC from mixed classes, 10 FASDD-RS test images). Colours are plausible in both. USTC is darker and lower in detail (mean RGB of the whole dataset 78 / 75 / 73, std 59 / 57 / 56; the 10 FASDD-RS images 92 / 110 / 99, std 41 / 40 / 40), and its content is large-scale (cloud banks, dust and haze fields, coastlines at 1 km per pixel) while FASDD-RS shows high-resolution land with small plumes. This is the resolution, sensor and region shift stated in Section 9, not a loading error, so the loader is unchanged. |

If the images are 16-bit or the channel order differs from FASDD-RS, the loader is fixed **before** any prediction is run,
and the fix is described here. Neither was the case, so no loader change was made.

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

**Primary (main result).** Identical to the FASDD evaluation (`build_eval_transform` in `src/firecls/transforms.py`):
resize the shorter side to 224 px, centre-crop 192 × 192, ImageNet mean and standard deviation. For a 256 × 256 USTC
image this keeps the central 192 × 192 px of a 224 × 224 resize, i.e. the central 73% of the image area (192² / 224²);
FASDD-RS images go through the same transform.

**Secondary (sensitivity analysis).** Resize the full 256 × 256 image directly to 192 × 192 (`transforms.Resize((192, 192))`,
torchvision default interpolation), with no 224 resize and no centre crop, then the same tensor conversion and ImageNet
mean and standard deviation. This keeps the whole scene, which matters because a smoke plume near the border of a USTC
image is cut off by the primary transform. Both settings are reported; the primary setting stays the main result.

## 5. Label mapping and prediction rule

| USTC label | Binary ground truth |
|---|---|
| `smoke` | positive |
| `cloud`, `dust`, `haze`, `land`, `seaside` | negative |

- **Primary prediction rule (main result):** an image is predicted positive if the predicted class (argmax of the four
  logits) is `fire`, `smoke` or `both`; it is predicted negative if the predicted class is `neither`.
- **Secondary prediction rule (sensitivity analysis):** an image is predicted positive only if the predicted class is
  `smoke` or `both`; `fire`, and `neither`, count as negative. Active fire is rarely visible in MODIS true-colour imagery
  at 1 km, so this checks whether the primary rule is inflated by `fire` predictions. Both rules are reported; the
  primary rule stays the main result.
- **Threshold-free score (ROC-AUC only):** under the primary rule `1 − P(neither)`; under the secondary rule
  `P(smoke) + P(both)` (that is `1 − P(neither) − P(fire)`), from the softmax of the logits.
- No threshold, temperature or calibration is tuned on USTC_SmokeRS.

## 6. Metrics

For every model, under each combination of preprocessing (Section 4) and prediction rule (above); the four combinations
are always computed together and only (primary preprocessing, primary rule) is the main result:

1. Smoke recall: share of `smoke` images predicted positive.
2. False-alarm rate: share of negative images predicted positive, overall and separately for `cloud`, `dust`, `haze`,
   `land` and `seaside`.
3. Balanced accuracy: mean of smoke recall and (1 − overall false-alarm rate). It is reported next to the per-class
   results because the classes differ in size (1,002 to 1,164 images).
4. ROC-AUC of the threshold-free score of the rule in use.
5. Distribution of the four predicted classes on `smoke` images (share predicted `smoke`, `both`, `fire`, `neither`);
   it does not depend on the rule, only on the preprocessing.

## 7. Statistics

- 95% confidence intervals: bootstrap over images, 2,000 resamples, as for the FASDD results.
- Pairwise comparisons: exact two-sided McNemar test on binary correctness, for every pair of the five deployed models
  (10 pairs), with Holm correction across the 10 tests. Report b, c, the difference in balanced accuracy and both the
  unadjusted and Holm-adjusted p-values. The correction is applied within each of the four (preprocessing, rule)
  combinations separately; the main result is the primary combination.
- A non-significant difference is reported as inconclusive, not as evidence of equal performance.
- The two secondary settings are sensitivity analyses: they are described, not used to choose a model, a rule or a
  preprocessing.

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
- All 6,225 images are used as test images and there is no train/validation/test split. Images from the same smoke event
  or the same MODIS granule may be correlated, so the images are not fully independent and the bootstrap intervals
  (which resample images) can be slightly optimistic.
- Labels are one scene-level label per image from the dataset authors; they were not re-checked, and some `smoke`
  images show only faint plumes. The copy used is the Hugging Face mirror (Section 2), not the original Baidu Pan archive.

## 10. Deviations

Record any change made after the first prediction was run.

| Date | Change | Reason |
|---|---|---|
| | | |
