# YOLO baseline: YOLO11n reference detector

Branch `baseline/yolo11n` (formerly `baseline/yolo-vazquez`). Provenance marker in the results
table: **official**: YOLO11n is the unmodified Ultralytics implementation; only the image-level
decision rule and the training protocol around it are ours (§2).

## 1. What this baseline is, and why YOLO11n

* **Role.** The reference for one-stage detectors, the family most used in published FASDD
  experiments: the dataset authors report YOLOv5x (Wang et al., 2025); EFSD-YOLO and SFGI-YOLO
  (both 2025) are modified YOLOv11 models; Vazquez et al. (2025) use nano YOLO detectors for
  edge wildfire detection.
* **Why the base model and not a published variant.** EFSD-YOLO and SFGI-YOLO have no public
  code, so reimplementing their modules from the papers would make any gap ambiguous
  (method vs. reimplementation). Both report gains of about 1-2 points over their YOLOv11 base,
  so the base model represents the family closely. Custom modules may also not export cleanly
  to ONNX/TensorRT, which would confound the edge measurements.
* **Why nano.** It is the size intended for edge deployment, which is the thesis setting.
* **Configuration.** YOLO11n fine-tuned from the official MS COCO checkpoint, seeds 42/43/44.
  Unmodified Ultralytics training pipeline (augmentation, loss, schedule, EMA, checkpoint
  selection) with library defaults except for the protocol budget (§4).
* **Edge metrics.** Latency, throughput and, on Jetson, power, measured by the shared
  `scripts/benchmark_baseline.py` on the same runtimes (PyTorch, ONNX Runtime, TensorRT FP16)
  as the distilled student.
* **Detector quality.** Precision, recall, mAP50 and mAP50-95 per domain on the test split
  (`scripts/yolo/detection_metrics.py`), as a secondary table.

The earlier grid (YOLOv5nu / YOLOv8n / YOLO11n x COCO / scratch) is still available through
`ARCHES` and `INITS` (§5) but is not part of the thesis comparison.

## 2. What is adapted, and why

1. **Image-level decision rule.** The thesis task is 4-way image classification
   (`fire, smoke, both, neither`). For each image the detector's dense, pre-NMS output gives
   `p_fire = max_anchors P(fire)` and `p_smoke = max_anchors P(smoke)`. NMS only removes boxes and
   never raises the highest score, so these equal the confidence of the best surviving
   detection. The pair is mapped to the 4-way scores by `firecls.baselines.scores.PresenceToScores`
   with thresholds `(t_fire, t_smoke)` grid-searched on the **validation** split
   (objective: equal-weight mean macro-F1 over the CV, RS and UAV domains). The thresholds are
   baked into the ONNX graph, so the TensorRT engine reproduces the PyTorch decision.
2. **Data.** Training uses FASDD CV + RS + UAV `train` images, with the image-level splits fixed by
   the committed `data_index/*.csv` (identical to the student and every other baseline). Only
   the boxes come from the FASDD annotation files. The preparation script checks that the label
   implied by the boxes equals the CSV label, and fails above 1 % disagreement per domain.
3. **No FASDD-pretrained weights.** Publicly released FASDD-trained YOLO weights (e.g. those of
   Vazquez et al.) are deliberately not used, because they may have been trained on images in
   our test split (leakage). All runs start from the COCO checkpoint.
4. **Input geometry.** Detectors keep their native 640 × 640 letterbox input (0–1 RGB, grey
   padding 114, no mean/std). Down-scaling to the student's 192 px would disadvantage a
   detector, so latency is reported at each method's native input size. NMS is not part of the
   measured graph because the decision rule does not need it.
5. **Model selection (deviation from `docs/BASELINES.md`).** `best.pt` is chosen by Ultralytics'
   own fitness on the combined validation set. In ultralytics 8.4 the fitness is mAP50-95 (older
   releases used 0.1·mAP50 + 0.9·mAP50-95). It is **not** the protocol's equal-weight mean
   validation accuracy, and on the combined set CV images are about 78 % of the validation
   images. Changing this would require modifying the Ultralytics trainer, which would no longer
   be the unmodified library pipeline. The image-level thresholds are then
   calibrated on validation only. The test split is never used for any choice.
6. **(Optional grid only) YOLOv5n is `yolov5nu`.** Ultralytics distributes only the *anchor-free* YOLOv5 variant
   ("u": v5 backbone and neck with the v8 decoupled, DFL-based head). It differs from the original
   anchor-based YOLOv5n of the `ultralytics/yolov5` repository. State this wherever "YOLOv5n"
   appears in the thesis.
7. **Optimizer (fixed explicitly).** Ultralytics' `optimizer=auto` depends on the library
   version: with our data (61,323 training images, nominal batch 64, 20 epochs, about 19,180
   iterations, which is over 10,000), ultralytics 8.4.164 resolves it to **MuSGD**, whereas earlier
   releases resolved the same case to **SGD** (lr 0.01,
   momentum 0.9). `run_all.sh` therefore passes `--optimizer SGD --lr0 0.01 --momentum 0.9` by
   default, so results do not depend on the installed Ultralytics version. Override with
   `TRAIN_ARGS=...` if needed; the optimizer actually used is recorded in `train_summary.json`
   (`resolved_optimizer`).
8. **Preprocessing implementation.** Image-level evaluation uses the shared PIL letterbox
   (`firecls.baselines.preprocessing.Letterbox`, centred square padding) for every runtime.
   Ultralytics' own validation (detection table) uses its OpenCV letterbox with rectangular
   batches. The interpolation differs slightly between the two.
9. **Parameter counts** in manifests and benchmarks are counted after Conv+BN fusion. This is the
   deployed graph, and it has slightly fewer parameters than the training graph.

## 3. Code

| File | Role |
|---|---|
| `src/firecls/baselines/families/yolo.py` | family `yolo`: loader, `YoloPresenceScores` wrapper, threshold lookup |
| `scripts/yolo/prepare_fasdd_yolo.py` | Ultralytics dataset from CSV splits + FASDD boxes |
| `scripts/yolo/train_yolo.py` | one training run (arch × init × seed), resumable |
| `scripts/yolo/calibrate_thresholds.py` | `(t_fire, t_smoke)` on val → `thresholds.json` |
| `scripts/yolo/detection_metrics.py` | per-domain test P/R/mAP (Ultralytics `val`) |
| `scripts/yolo/run_all.sh` | full grid, skips finished steps |
| `tests/test_yolo_baseline.py` | CPU tests without data or network |

Verified behaviour of ultralytics 8.4.164: in eval mode `DetectionModel.forward` returns a tuple
`(decoded [B, 4+nc, 8400], raw)`. With `Detect.export = True` it returns the decoded tensor
alone, and its class channels are already sigmoid probabilities. The wrapper fuses Conv+BN and sets
the same head flags as Ultralytics' ONNX exporter, so the PyTorch module and the ONNX graph
compute the same thing (verified: ONNX vs PyTorch max abs error below 1e-6 at batch sizes 1, 2 and 5).

`thresholds.json` stores the sha256 of the checkpoint it was calibrated for. The loader rejects a
mismatch. If the file is missing, the loader falls back to 0.5/0.5, prints a warning and records
`"thresholds_calibrated": false` in the metadata.

## 4. Training configuration

Protocol values (identical for all baselines): 20 epochs, batch 16, seeds 42/43/44, input 640,
`deterministic=True`. Everything else is the Ultralytics 8.4.164 default, saved verbatim in
`args.yaml` and `train_summary.json`. The most relevant defaults:

| Setting | Default |
|---|---|
| optimizer | SGD, lr0 0.01, momentum 0.9 (see §2.7), weight decay 5e-4, warm-up 3 epochs, linear LR decay to `lrf = 0.01` |
| augmentation | mosaic 1.0 (switched off for the last 10 epochs), HSV (0.015, 0.7, 0.4), translate 0.1, scale 0.5, horizontal flip 0.5 |
| loss gains | box 7.5, cls 0.5, dfl 1.5; nominal batch 64 |
| other | AMP on CUDA, EMA of weights, patience 100 (no early stop within 20 epochs), workers 8 |

COCO initialisation transfers every tensor except the 80-class output convolutions (e.g. 391 of 427
tensors for YOLOv5nu), which are replaced by a 2-class head (`fire = 0`, `smoke = 1`).

## 5. Commands

```bash
pip install -r requirements-baselines.txt

# 1. dataset (once): box paths depend on how FASDD was unpacked (COCO .json or YOLO labels dir)
python scripts/yolo/prepare_fasdd_yolo.py \
    --cv-boxes <FASDD_CV COCO json> --rs-boxes <FASDD_RS COCO json> --uav-boxes <FASDD_UAV COCO json>
#    for YOLO label dirs without classes.txt add: --yolo-class-names fire smoke

# 2. one cell of the grid
python scripts/yolo/train_yolo.py --arch yolo11n --init coco --seed 42 --device 0
CKPT=outputs/baselines/yolo/yolo11n_coco/seed42/weights/best.pt
python scripts/yolo/calibrate_thresholds.py --checkpoint $CKPT
python scripts/export_baseline.py --family yolo --checkpoint $CKPT \
    --output artifacts/baselines/yolo11n_coco_seed42.onnx
python scripts/evaluate_baseline.py --family yolo --name yolo11n_coco_seed42 --checkpoint $CKPT \
    --onnx artifacts/baselines/yolo11n_coco_seed42.onnx \
    --output results/baselines/yolo11n_coco_seed42_eval.json
python scripts/yolo/detection_metrics.py --name yolo11n_coco_seed42 --checkpoint $CKPT \
    --output results/baselines/yolo11n_coco_seed42_detection.json
python scripts/benchmark_baseline.py --family yolo --name yolo11n_coco_seed42 --checkpoint $CKPT \
    --onnx artifacts/baselines/yolo11n_coco_seed42.onnx \
    --output results/baselines/yolo11n_coco_seed42_bench.json

# or all three seeds of YOLO11n (COCO init), resumable:
CV_BOXES=... RS_BOXES=... UAV_BOXES=... DEVICE=0 scripts/yolo/run_all.sh
# optional earlier grid, not used in the thesis:
ARCHES="yolov5n yolov8n yolo11n" INITS="coco scratch" scripts/yolo/run_all.sh
BUILD_ENGINE=1 scripts/yolo/run_all.sh     # additionally build + evaluate TensorRT FP16 engines
```

TensorRT engines and Jetson measurements follow `docs/BASELINES.md` §3 (the engine is built on
the target device from the verified ONNX).

Network access during training: the COCO checkpoints are downloaded once into
`third_party/weights/` (gitignored). Ultralytics also downloads `Arial.ttf` for its plots and, on
CUDA, a small `yolo26n.pt` for its AMP self-check (not used for training). On an offline machine,
place these files in advance.

## 6. Expected outputs

```
datasets_yolo/                                       (gitignored)
  images/<domain>/<split>/<domain>_<stem>.<ext>      symlinks to FASDD images
  labels/<domain>/<split>/<domain>_<stem>.txt
  <domain>_<split>.txt, fasdd_all.yaml, fasdd_<domain>.yaml, prepare_report.json
outputs/baselines/yolo/<arch>_<init>/seed<k>/        (gitignored)
  weights/best.pt, weights/last.pt, weights/thresholds.json
  args.yaml, results.csv, train_summary.json, Ultralytics plots
artifacts/baselines/<arch>_<init>_seed<k>.onnx(.manifest.json)
results/baselines/<arch>_<init>_seed<k>_{eval,detection,bench}.json
```

Row names follow `<arch>_<init>_seed<k>`, e.g. `yolo11n_coco_seed42`. They are aggregated over seeds as
mean ± sample std by `scripts/collect_results_table.py`.

## 7. Methods paragraph (LaTeX)

```latex
\paragraph{Detector baseline.}
One-stage YOLO detectors are the approach most frequently evaluated on FASDD: the dataset
authors report results for YOLOv5x~\cite{wang2025fasdd}, EFSD-YOLO~\cite{efsdyolo2025} and
SFGI-YOLO~\cite{sfgiyolo2025} are modified YOLOv11 models, and \citet{vazquez2025} employ nano
YOLO detectors for edge-based wildfire detection. YOLO11n~\cite{ultralytics_yolo11} is used as the
representative of this family. It is the common base of the recent FASDD-specific variants, for
which no public implementation is available and which report gains of one to two points over
their base model; reproducing their modules from the publications alone would make any
difference in performance difficult to attribute. The nano variant was chosen because it is the
model size intended for edge deployment. YOLO11n was fine-tuned from the official MS~COCO
weights on the fire and smoke bounding boxes of the FASDD computer-vision, remote-sensing and UAV
subsets, restricted to the training images of the fixed image-level partition used for all
methods in this work, for 20 epochs with a batch size of 16 at its native input resolution of
$640 \times 640$ pixels; all remaining hyper-parameters follow the defaults of Ultralytics~8.4.164,
except that stochastic gradient descent (learning rate $0.01$, momentum $0.9$) was set explicitly.
Publicly released FASDD-trained weights were not used, as they may have been exposed to images of
the test partition. Because the present task is image-level classification, the detector is
converted into a four-way classifier by a deterministic decision rule: the presence probability of
each class is the maximum class confidence over all candidate boxes prior to non-maximum
suppression, and an image is assigned to \emph{fire}, \emph{smoke}, \emph{both} or \emph{neither}
by comparing the two presence probabilities with class-specific thresholds. The thresholds are
selected by grid search on the validation partition to maximise the macro-averaged $F_1$ score,
averaged with equal weight over the three domains, and are embedded in the exported inference
graph. Checkpoint selection relies on the detection fitness computed by the Ultralytics trainer
on the combined validation set; the test partition is used exclusively for the final evaluation.
The detector is trained with three random seeds, and results are reported as mean and sample
standard deviation.
```

The citation keys are placeholders. Verify the EFSD-YOLO and SFGI-YOLO bibliographic details
and the Ultralytics YOLO11 software citation before finalising the text.
