# Fine-tuning with additional Sentinel-2 smoke data: protocol

Branch `retrain/rs-smoke-plumes` (from `origin/retrain-pipeline` at `b6b9188`). These decisions are fixed **before any
training run**: the commit that adds this file is the reference point. Any later change goes into the Deviations table
(Section 10) with date and reason. Nothing here is chosen by looking at fine-tuning results.

Status: all sections are final. The RGB conversion (Section 4) was approved by Amir on 2026-10-23 (per-channel version) and is
frozen. Items marked **[doubt]** are choices the implementing agent had to make and that Amir may overrule.

## 1. Task framing

- **RS is evaluated as a binary task: smoke/fire vs neither.**
  - Ground truth positive: FASDD-RS `smoke` or `both`; plume. Negative: FASDD-RS `neither`; non-plume.
  - Prediction positive: predicted `fire`, `smoke` or `both`. Negative: predicted `neither`.
  - Threshold-free score: `1 − P(neither)` (softmax of the four logits).
- Rationale (Chapter 3): FASDD-RS has no `fire`-only image in any split (train 0, val 0, test 0); RS fire always appears with
  smoke and is labelled `both`.
- **Training keeps the 4-class scheme** for every model: plume → `smoke`, non-plume → `neither`.
- CV and UAV are evaluated as before (4 classes).
- The current ("before") models' RS results are recomputed in binary form from existing predictions. They were recomputed with
  `scripts/evaluate_rs_binary.py` (RS teacher: PyTorch FP32 dump made on the main machine for this purpose, because no teacher
  prediction file exists on any branch; the five others: the TensorRT FP16 dumps of the `results` branch at `d96b64a`) and
  agree with the reference values of the decisions document in every cell:

| Model | Recall % | False-alarm % | Balanced acc. % | F1 % | ROC-AUC | Errors / 370 | Recall on 31 `smoke` % | Recall on 191 `both` % |
|---|---|---|---|---|---|---|---|---|
| RS teacher (FP32) | 97.3 | 5.4 | 95.9 | 96.9 | 0.9921 | 14 | 90.3 | 98.4 |
| KD routed | 95.5 | 4.7 | 95.4 | 96.1 | 0.9845 | 17 | 93.5 | 95.8 |
| KD uniform | 92.8 | 7.4 | 92.7 | 93.8 | 0.9655 | 27 | 90.3 | 93.2 |
| No KD | 94.6 | 9.5 | 92.6 | 94.2 | 0.9643 | 26 | 93.5 | 94.8 |
| MobileNetV3-Large | 93.2 | 7.4 | 92.9 | 94.1 | 0.9691 | 26 | 90.3 | 93.7 |
| ResNet-50 | 91.9 | 8.1 | 91.9 | 93.2 | 0.9666 | 30 | 87.1 | 92.7 |

  (Seed 42; students and baselines are TensorRT FP16. The two dumps list the 370 images in different orders; the evaluator
  aligns them by file name and checks that the labels agree.)
- **Role of each test set** (fixed in advance): FASDD-RS test = non-inferiority check ("does fine-tuning do no harm?"; the models
  are near the ceiling, 14 to 30 errors, so a significant gain is not expected); Plume test groups = measures the gain from the
  added data; FASDD CV and UAV tests = forgetting check.

## 2. Data source

- Mommert, M., Sigel, M., Neuhausler, M., Scheibenreif, L., Borth, D. (2020). Characterization of Industrial Smoke Plumes from
  Remote Sensing Data. NeurIPS 2020 Workshop Tackling Climate Change with ML. Zenodo record 4250706, v1.0 (CC BY 4.0),
  https://zenodo.org/records/4250706; arXiv 2011.11344. Downloaded 2026-10-23.
- `images.tar.gz` (6,359,411,128 bytes): MD5 `d87992cf622920dc0135e16bac9e2cfd` (matches Zenodo), SHA-256
  `808e839b98e247fdd613b29d63796bde1af08eed7b6a630d8dc39a21fd5ba71c`. `README.md` (MD5 `859d07bb…`) and
  `segmentation_labels.tar.gz` (MD5 `73e6124b…`) also fetched; MD5 match.
- **The label is the folder inside the tar** (`positive/` 3,750, `negative/` 17,600), not the file name. `scripts/plumes/
  extract_plume_archive.py` keeps the folders. A first extraction that flattened them was deleted; the second was verified
  byte-identical (SHA-256) to it.
- 21,350 images, 624 site ids (the leading number of the file name), all from 2019; 423 sites with plumes, 353 with both
  labels, 70 with plumes only. GeoTIFF, 13 bands, uint16, reflectance × 10,000, UTM, 10.0 × 10.0 m pixels, 120 × 120 px
  (24 images 120 × 119). RGB = B4, B3, B2 (bands 4, 3, 2 of the file; band order B1…B8, B8A, B9, B10, B11, B12, confirmed by
  band 11 = B10 cirrus having very small values).
- Local path `datasets/IndustrialSmokePlumes/` (`raw/`, `rgb/`), never committed (ignored by `datasets/` and `*.tif`).
- Segmentation polygons are not used for training or evaluation (only as an anchor for the label-noise check, Section 6).

## 3. Selection and split

Implemented by `scripts/plumes/select_plumes.py` (seed 42, deterministic: a rerun gives an identical file) from the table written
by `scripts/plumes/site_table.py`.

- **3,000 plume + 3,000 non-plume**, split **2,100 / 450 / 450 per class** for train / val / test (kept as is, confirmed by Amir
  on 2026-10-23). **Groups per split: 273 train, 118 val, 124 test** (562 groups in the whole data; 515 of them hold at least one
  selected image).
- **Split unit = group:** site ids whose image centres are within 2.5 km (image diagonal 1.7 km plus centre jitter) are merged;
  562 groups, largest 4 ids. No group appears in two splits (checked).
- **Plume images:** all 322 groups with both plume and non-plume images are used whole (2,587 images), then plume-only groups are
  added in random order until 3,000, the last one trimmed at random. **413 plume images come from 19 plume-only groups** (of 57
  available); they are **not** spread proportionally over the splits: train 365 images (13 groups, 17% of its plume images),
  val 15 (2 groups, 3%), test 33 (4 groups, 7%). Consequence: the test set has almost no plume image without same-group
  negatives, the training set has more. (The decisions document said "70 plume-only groups, distributed proportionally"; 70 was
  the number of plume-only site ids, and the assignment follows the seeded largest-first fill, not proportions.)
- **Non-plume images:** one distinct non-plume image per plume image inside each split, by optimal assignment (cost = 2 × calendar
  month gap + 1 if from another group). 2,988 of 3,000 pairs share a month; 1,005 non-plume images (34%) share the group of their
  plume image. Reason: plumes are winter-heavy (41–42% of all images in Dec–Jan, 9–10% in Jul–Aug); a "winter = smoke" cue would
  hurt on FASDD-RS, whose wildfire scenes are mostly summer, while a site cue cannot help on unseen test groups. A 0.5 : 1
  weighting would give 49% same-group but only 83% same-month pairs.
- **Report (from the index):**

| Split | Plume | Non-plume | Groups | Site ids | Plume-only groups (plume images from them) |
|---|---|---|---|---|---|
| train | 2,100 | 2,100 | 273 | 311 | 13 (365) |
| val | 450 | 450 | 118 | 124 | 2 (15) |
| test | 450 | 450 | 124 | 139 | 4 (33) |

  Month histograms (Jan–Dec), plume | non-plume: train `155 264 180 251 92 145 127 160 156 200 135 235` for both; val
  `83 65 23 28 13 21 13 13 13 47 39 92` | `71 74 23 28 13 21 13 13 13 47 39 95`; test `63 66 32 35 12 15 8 8 18 39 56 98` for
  both.
- 24 images of 120 × 119 px are excluded (1 plume, 23 non-plume). 12 selected images are entirely black (pixel maximum 0 after conversion; all non-plume: 10
  train, 1 val, 1 test); they are kept, a black tile being trivially "neither".
- Committed as `data_index/industrial_plumes.csv` (`path,site,group,month,label_src,label,split`; SHA-256
  `b9c8fe574979e76bdd433bbcdd8aff7ca5da8e0c6460ab7954b873451d59455c`); the CSV is the list of the 6,000 selected files, and
  `data_index/industrial_plumes.meta.json` holds the report above.

## 4. RGB conversion (approved and FROZEN, 2026-10-23)

- Base conversion: **gain 2.0, gamma 0.6**: `rgb = clip(reflectance × gain, 0, 1)^gamma × 255` with `reflectance = value / 10,000`
  (best of a grid of 8 gains × 8 gammas, mean absolute difference 15.30 in 8-bit units to the FASDD-RS **training** pixel
  percentiles 1, 5, 10, 25, 50, 75, 90, 95, 99 per channel; the pre-set criterion). Next best: gain 2.25 / gamma 0.65, 15.41.
- **Final conversion = the per-channel version** (decision of the decisions document, fitted on 1,000 plume **train** images and
  the FASDD-RS **train** split only; seeds 0 and 1 of `choose_rgb_conversion.py`):

  | Parameter | Value |
  |---|---|
  | bands | B4, B3, B2 = bands 4, 3, 2 of the GeoTIFF (R, G, B) |
  | reflectance | `value / 10,000` |
  | gain, gamma | 2.0, 0.6: `z_c = (reflectance_c × 2.0)^0.6` (no clipping) |
  | per-channel gain `k` (R, G, B) | 1.0108248811066958, 1.1724013075856523, 1.3241686092164435 |
  | FASDD-RS train medians matched (R, G, B) | 85, 97, 95 |
  | highlight shoulder `y0` | 0.3813921568627451 |
  | output | `out_c = round(255 · clip(h(k_c · z_c), 0, 1))`, `h(y) = y` for `y ≤ y0`, else `y0 + (1 − y0)(1 − exp(−(y − y0)/(1 − y0)))` |
  | p99 target of the shoulder | 220 (FASDD-RS train p99: 200, 212, 221) |
  | score vs FASDD-RS train percentiles (400-image sample of the selected 6,000, seed 0) | **8.56** (base 15.30) |
  | share of pixels at 255 | 0 (base 1.8%) |

  Rule: adopted only if not worse than the base score → **adopted**. **[doubt]** "cap p99 at about 220" was read as this
  one-parameter shoulder; the lowest allowed `y0` (just above the green median) was reached, and the achieved p99 is 213, 224,
  232 per channel, so the target is only approximately met.
- Written once by `scripts/plumes/convert_to_rgb.py` as **lossless 8-bit PNG** (120 × 120, RGB, uint8) to
  `datasets/IndustrialSmokePlumes/rgb/<positive|negative>/<name>.png` (3,000 + 3,000 files); the existing loader
  (`PIL.Image.open(...).convert("RGB")`) is reused unchanged. `rgb/conversion.json` holds the parameters and the SHA-256 of every
  file (its own SHA-256: `55195254fe947796640b161ddfe3405c208ccd72f72acf6a1fba2f7c0bca0d6c`); the same per-file hashes are in
  `data_index/industrial_plumes_rgb.sha256` (SHA-256 `0c2bc2cc30e47a60b919bc4ba5fe44d6bd0fe99b11d3d1b6374a2c35dc646046`), so any
  machine can verify its copy. The files were set read-only; the converter refuses to overwrite. **No later changes.**
- The files are byte-identical (6,000 of 6,000) to the provisional scratch conversion used for the code tests and for the
  label-noise review, so that review applies to the frozen images as they are.

## 5. Image size and scale (record only)

- FASDD-RS: 1000 × 1000 px; pixel spacing 8.9832e-5° in both directions, i.e. 10.00 m north-south and 3.4–10.0 m east-west
  (median 9.7 m, because the grid is square in degrees); footprint about 10 × 9.7 km. Plume: 120 × 120 px, 10 × 10 m (UTM);
  footprint 1.2 × 1.2 km. FASDD-RS is on a 10 m grid; the metadata cannot show the native resolution of its source sensor.
- Effective size of one model-input pixel (evaluation: resize to 224, centre-crop 192): **FASDD-RS about 45 m, plume about 5.4
  m; plumes appear about 8.3× larger** in the model input.
- **Transforms are not changed.** Shrinking plume images to the FASDD-RS scale would make them about 27 px wide and remove the
  plume.
- **Georeferencing of the 2,223 images in `data_index/fasdd_rs_tdml.csv`:** 332 (15%) have no CRS (train 172, val 105, test 55); the
  other 1,891 have a 10.00 m north-south pixel and an east-west pixel of 3.39–10.00 m (median 9.66 m). The 332 are assumed to
  share the same spacing. The folder holds 3,558 files; the 1,335 not in the index are all georeferenced. Two indexed images are
  1000 × 999 px (`smoke_RS000472.tif` test, `smoke_RS000473.tif` train).

## 6. Data-quality checks (record only)

- Sharpness (variance of the Laplacian of the 192 × 192 evaluation view of the frozen PNGs; plume minus non-plume in pooled SDs;
  `scripts/plumes/check_sharpness.py`): all **−0.016** (means 52.2 and 52.7), train −0.040, val +0.157, test −0.102. (A first
  measurement on a grid-search conversion gave +0.02 / −0.02 / +0.25 / −0.03; the frozen conversion is the one that counts.) No
  re-split.
- **Label noise (blind review of 100 plume + 100 non-plume TRAINING images, seed 0; reviewer: the implementing agent, judging at
  full field of view with the per-channel rendering, pixel-identical to the frozen images; verdicts saved before the key was
  opened; `scripts/plumes/label_noise_sheets.py`):**

| Source label | n | clear plume | faint / uncertain | none visible |
|---|---|---|---|---|
| plume | 100 | 41 | 31 | 28 |
| non-plume | 100 | 2 | 12 | 86 |

  Anchor: 41 of the 100 plume images have a manual polygon from the dataset authors, yet the reviewer judged 13 of them "none";
  so the reviewer misses about a third of plumes the annotators outlined, and the "none visible" rate (28%) is an **upper bound** of
  the label noise, not an estimate of it. Nothing is relabelled or removed. Cloud-covered non-plume images are kept (hard
  negatives for cloud vs smoke). Made on images byte-identical to the frozen ones (Section 4), so it needs no repetition.

## 7. Models and fine-tuning

| Model | Starts from | Data | Loss |
|---|---|---|---|
| RS teacher (SwinV2-L) | `outputs/teachers/FASDD_RS/best.pt` | FASDD-RS train + plume train | cross-entropy |
| Student, KD routed | seed-42 `best.pt` | FASDD train (CV + UAV + RS) + plume train | original KD loss; plume soft targets from the **fine-tuned** RS teacher |
| Student, KD uniform | seed-42 `best.pt` | same | original KD loss; all three teachers score plume images (RS teacher = fine-tuned) |
| Student, no KD | seed-42 `best.pt` | same | cross-entropy |
| MobileNetV3-Large | seed-42 checkpoint | same | original loss |
| ResNet-50 | seed-42 checkpoint | same | original loss |

Out of scope: YOLO11n (deferred), USTC_SmokeRS (dropped; the `external/ustc-smokers` branch is left unmerged).

Implementation: `scripts/finetune_rs_plumes.py` with `src/firecls/finetune/` (data, recipes, runner), `src/firecls/data/plumes.py`,
`src/firecls/binary_eval.py`, `scripts/evaluate_rs_binary.py`, `tests/test_plume_finetune.py`.

- **Plume images are tagged as the RS domain** (RS domain id 1 in `cv, rs, uav` order) in the training set, so routed distillation
  takes their soft target from the RS teacher; unit test `test_plume_samples_get_rs_domain_id_and_rs_teacher_soft_target`.
- **Order:** RS teacher first; the KD students load the fine-tuned RS teacher (`--rs-teacher`).
- **Settings:** each model keeps its original optimizer, augmentation, batch size and AMP setting; **5 epochs at one tenth of its
  original learning rate**; schedules restarted over the 5 epochs; one epoch = one pass over the model's full fine-tuning data.
  Fresh AdamW state (the old optimizer state is not loaded) **[doubt]**.

| Model | lr | weight decay | schedule | clip | AMP | batch | source of the values |
|---|---|---|---|---|---|---|---|
| RS teacher | 3e-6 | 0.01 | constant | none | yes | 16 | `train_teacher.py` defaults; **batch size and AMP were not recorded** **[doubt]** |
| KD students | 3e-6 | 0.01 | constant | 1.0 | no | **8** | `train_student.py`; batch 8 (not the script's 16) from the logs: 7,666 iterations per epoch = 61,323 / 8 |
| No-KD student | 3e-6 | 0.01 | constant | 1.0 | yes | **8** | `train_classifier.py`; logs: 7,665 iterations (drop_last) |
| MobileNetV3, ResNet-50 | 3e-5 | 0.05 | cosine, 1 warm-up epoch | 1.0 | yes | 16 | `history.json` args; the warm-up is kept at 1 of the 5 epochs **[doubt]** |

  Peak GPU memory measured on the RTX 3070 (8 GB) in dry runs: teacher 6.9 GB, KD student with three teachers 5.8 GB, no-KD
  student 2.6 GB; all fit with these settings.
- **Sampler:** normal. RS training becomes about 2,770 positive and 2,540 negative.
- **Checkpoint selection:** highest mean of (CV 4-class val accuracy, UAV 4-class val accuracy, RS binary val balanced accuracy),
  with RS val = FASDD-RS val + plume val pooled; the two RS parts are logged separately every epoch, validation is FP32. Epoch 0
  (the before model) is evaluated and logged but never selected. The RS teacher has no CV / UAV data, so its score is the pooled RS
  binary balanced accuracy alone **[doubt]**. The control runs use the same rule, including plume val **[doubt]**.
- **Pre-registered fallback:** if binary recall on the FASDD-RS **val** `both` images of the selected checkpoint drops by more than
  2 points from the before model (epoch 0), rerun that model with the class-balanced sampler for the RS portion
  (`--balanced-rs-sampler`: every four-class label of FASDD-RS train + plume train gets equal weight, the RS portion keeps its total
  weight, CV and UAV images weight 1) and report both runs. `summary.json` of every run records the drop and whether it triggered.
- **Second pre-registered fallback, RS teacher only (source balance):** images of the plume dataset (2,100 plume + 2,100
  non-plume) are 79% of the teacher's RS training data (4,200 of 5,312 images; FASDD-RS train has 1,112). If the fine-tuned
  teacher's **binary balanced accuracy on FASDD-RS val** (selected checkpoint) drops by **more than 2 points** from the before
  teacher (epoch 0), the teacher is rerun with **source-balanced batches**: every batch of 16 has exactly 8 FASDD-RS train and 8
  plume-dataset train images (`--source-balanced-batches`; one epoch = 332 batches of 16 = 5,312 images, the size of the
  unbalanced epoch; each source is drawn without replacement from a shuffled queue that is reshuffled when exhausted, so FASDD-RS
  is passed about 2.4 times and the plume dataset about 0.6 times per epoch). Both runs are reported. `summary.json` records the
  drop (`fasdd_rs_val_ba_drop_points`) and the flag (`ba_fallback_triggered`).
- **KD students only after this check:** the KD students (routed and uniform) are fine-tuned only after the RS teacher check, with
  the first teacher run that passes it: the normal run if its drop is at most 2 points; otherwise the source-balanced rerun if
  that passes; if neither passes, stop and ask Amir. (The `both`-recall fallback above is also evaluated for the teacher and
  reported; if it triggers for the teacher, stop and ask Amir rather than start a third teacher run **[doubt]**.) The teacher
  checkpoint used is recorded in each student's `meta` (`rs_teacher_sha256`).
- **Control:** for the routed student and MobileNetV3-Large, the same fine-tuning **without** plume images (`--no-plumes`).
- **Seeds:** 42 for all; 43 and 44 later for the routed student and MobileNetV3-Large at least. (Checkpoints present locally: KD
  routed and KD uniform seeds 42 and 43; no-KD seed 42; MobileNetV3-Large and ResNet-50 seed 42 only; the seed-44 directories hold
  only a `DONE` marker. Seeds 43 and 44 of MobileNetV3-Large would first have to be trained.)
- Smoke test first: MobileNetV3-Large, 1 epoch; losses decrease and predictions are not all one class.

## 8. Evaluation

- Accuracy from **TensorRT FP16 engines** built on the Jetson (latency not re-benchmarked; architectures unchanged). Teacher:
  PyTorch FP32. Predictions: `scripts/pipeline/02_dump_predictions.py` (`--domains cv rs uav plume`; `--family cnn` for the CNN
  checkpoints) and `scripts/evaluate_rs_binary.py`.
- Test sets and metrics, for every model before / after / control:

| Test set | Metrics |
|---|---|
| FASDD-RS test (370: 222 positive, 148 negative) | binary recall, false-alarm rate, balanced accuracy, F1, ROC-AUC; recall separately on the 31 `smoke` and 191 `both` images |
| FASDD CV and UAV test | 4-class accuracy and macro-F1 |
| Plume test groups (900: 450 / 450) | binary recall, false-alarm rate, balanced accuracy, F1, ROC-AUC |

- Statistics: bootstrap 95% CIs (2,000 resamples); exact McNemar on binary correctness for before vs after and for with vs without
  plumes, Holm-corrected within each test set. Non-significant results are reported as inconclusive.
- Outputs: `results` branch, `rs_plume_finetune/`; each `meta.json` records the commit hash and checkpoint / engine SHA-256.
- Tables and figures: (1) RS binary before/after on FASDD-RS; (2) 2 × 2 RS confusion matrices before/after; (3) CV and UAV
  before/after; (4) plume test-group results; (5) control comparison.

## 9. Limitations (for the thesis)

- Industrial point-source smoke, not wildfire smoke; all images from 2019.
- Plumes appear about 8.3× larger in the model input than smoke in FASDD-RS (Section 5); upsampled 120 px images.
- FASDD-RS east-west pixel size varies with latitude (3.4–10.0 m); 332 of its 2,223 indexed files lack georeferencing.
- Source-label noise (Section 6; upper bound 28% of plume images without a visible plume to the reviewer).
- Plume-only groups are 17% of the training plumes but 7% of the test plumes (Section 3).
- Single seed for most models; YOLO11n not included; the FASDD-RS test can only show non-inferiority, not a gain.

## 10. Deviations

| Date | Change | Reason |
|---|---|---|
| 2026-10-23 | 24 images of 120 × 119 px excluded | non-standard size |
| 2026-10-23 | `:` in file names replaced by `_` on disk; originals in `raw/images/manifest.csv` | Windows file-name restriction |
| 2026-10-23 | Split unit changed from site id to 2.5 km site group | nearby site ids cover the same location |
| 2026-10-23 | Plume-only groups: 19 groups, not 70, and not proportional over the splits (365 / 15 / 33 images) | 70 was a count of site ids; assignment follows the seeded fill |
| 2026-10-23 | Student batch size 8 (not 16) in the recipe table | recovered from the training logs |
| 2026-10-23 | Teacher fine-tuning: batch 16 with AMP | original batch size and AMP not recorded; the script default is the best available guess |
| 2026-10-23 | "Cap p99 at about 220" implemented as a one-parameter highlight shoulder; target only approximately met | see Section 4 |
| 2026-10-23 | Added the RS-teacher source-balance fallback and the rule that the KD students are fine-tuned only after the teacher check | requested by Amir: plume images are 79% of the teacher's RS training data |

## 11. Reproduction notes

- Tools that read the GeoTIFFs (`extract_plume_archive.py` needs only the standard library; `inspect_plumes.py`, `site_table.py`,
  `choose_rgb_conversion.py`, `convert_to_rgb.py` need `rasterio`) were run in a throwaway environment: `rasterio` is not a project
  dependency and training needs nothing beyond the existing requirements.
- Order of the data steps: `extract_plume_archive.py` → `site_table.py` → `select_plumes.py` → `choose_rgb_conversion.py` →
  `convert_to_rgb.py` (once) → `label_noise_sheets.py`, `check_sharpness.py`.
- Dataset root used for all runs (`--dataset-root`): a folder whose `datasets/` holds `FASDD_CV` (the CV images, here the folder
  `Fire_Detection_CV`), `FASDD_RS` (`Fire_Remote_Sensing`), `FASDD_UAV` and `IndustrialSmokePlumes`, so `firecls.config`'s paths and
  the index paths (`datasets/IndustrialSmokePlumes/rgb/...`) resolve; the indexes come from `data_index/` of this branch.
- Checkpoints are in the main checkout's `outputs/` (ignored by git); pass absolute paths.
