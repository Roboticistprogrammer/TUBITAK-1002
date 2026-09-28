# Baseline 3: CNN classifiers (ResNet-50, MobileNetV3-Large) and the no-KD ablation

Branch: `baseline/cnn-classifiers` (built on `baselines/common`; protocol in `docs/BASELINES.md`).

## Rows produced

| Row | Architecture | Initialisation | Role in the thesis |
|---|---|---|---|
| `resnet50` | torchvision ResNet-50 (23.5 M params with the 4-way head) | ImageNet-1k V2 | accuracy-oriented CNN reference (He et al., 2016) |
| `mobilenet_v3_large` | torchvision MobileNetV3-Large (4.2 M params with the 4-way head) | ImageNet-1k V2 | edge-oriented CNN reference (Howard et al., 2019) |
| `swinv2_base_nokd` | the student's SwinV2-B, identical recipe, cross-entropy only | ImageNet-22k (Hugging Face) | **ablation**: isolates the contribution of multi-teacher KD |

The `swinv2_base_nokd` row is the most important one for the thesis argument: it differs from
the proposed student *only* in the training objective (no teacher soft targets), so the gap
between the two is the effect of distillation.

## Why classifiers and not Faster R-CNN

Wrapping either backbone in Faster R-CNN is technically easy: torchvision ships
`fasterrcnn_resnet50_fpn_v2` and `fasterrcnn_mobilenet_v3_large_fpn`, and training on FASDD's
boxes would take roughly a hundred lines. It was not used, for three reasons:

1. **Task mismatch.** The proposed model is an image-level classifier. A detector answers a
   different question (where are the boxes), so its output would have to be reduced to an
   image label by a decision rule, and the comparison would then mix backbone quality with
   detector design. The classifier variant compares backbones like for like.
2. **Deployment.** Faster R-CNN's RoIAlign, anchor generation and NMS stages export to ONNX
   with data-dependent shapes. TensorRT needs plugins or graph surgery for them, which is
   fragile on JetPack 4 / TensorRT 8.2 (Jetson Nano). It would be a comparison of export
   workarounds rather than of methods.
3. **Coverage.** The detection paradigm is already represented by the YOLO baseline
   (`baseline/yolo-vazquez`), which is the detector family actually used for edge wildfire
   detection in the reviewed literature.

The thesis can state this in one sentence (see the paragraph below).

## Recipe

Shared with every baseline (from `firecls.baselines.protocol`): CV+RS+UAV `train` splits
concatenated; `RandomResizedCrop(192, scale=(0.7, 1.0))` + horizontal flip (identical to
`train_student.py`); evaluation resize 224 → centre-crop 192; 20 epochs; batch 16; selection by
best equal-weight mean validation accuracy; seeds 42, 43, 44.

Architecture-specific optimiser (recorded in every checkpoint under `recipe`):

| | CNNs | SwinV2-B no-KD |
|---|---|---|
| Optimiser | AdamW, lr 3e-4, weight decay 0.05 | AdamW, lr 3e-5, weight decay 0.01 (student's) |
| Schedule | 1 epoch linear warm-up, cosine decay | constant (student's) |
| Gradient clipping | 1.0 | 1.0 (student's) |
| Mixed precision | on (`--amp`) | on (`--amp`) |

**Tuning parity (recommended).** Run `SWEEP=1 scripts/cnn/run_all.sh` once. It trains each CNN
with lr ∈ {1e-4, 3e-4, 1e-3} on seed 42 and prints the best validation accuracy of each. Pick
the best lr per architecture and pass it with `LR_ARGS="--lr <value>"`. Report the grid in the
thesis. The student's lr was not tuned beyond its default, so do not tune the CNNs more than
this.

## Commands

```bash
pip install -r requirements.txt            # torch/torchvision/transformers already cover this branch

# everything (train -> export -> evaluate -> benchmark), resumable:
scripts/cnn/run_all.sh
BUILD_ENGINE=1 scripts/cnn/run_all.sh      # additionally TensorRT FP16 on this machine

# a single run:
python scripts/cnn/train_classifier.py --arch resnet50 --seed 42 --amp
python scripts/export_baseline.py --family cnn \
    --checkpoint outputs/baselines/resnet50/seed42/best.pt \
    --output artifacts/baselines/resnet50_seed42.onnx
python scripts/evaluate_baseline.py --family cnn --name resnet50_seed42 \
    --checkpoint outputs/baselines/resnet50/seed42/best.pt \
    --onnx artifacts/baselines/resnet50_seed42.onnx \
    --output results/baselines/resnet50_seed42_eval.json

# no-KD ablation uses the existing "swinv2" family for export/evaluation:
python scripts/cnn/train_classifier.py --arch swinv2_base_nokd --seed 42 --amp
python scripts/export_baseline.py --family swinv2 \
    --checkpoint outputs/baselines/swinv2_base_nokd/seed42/best.pt \
    --output artifacts/baselines/swinv2_base_nokd_seed42.onnx
```

On the Jetson: copy the `.onnx` files and use `scripts/jetson_trtexec_benchmark.sh <onnx> 192 1`.

Expected outputs per run: `outputs/baselines/<arch>/seed<k>/{best.pt,last.pt,history.json}`,
`artifacts/baselines/<arch>_seed<k>.onnx` + `.manifest.json`,
`results/baselines/<arch>_seed<k>_{eval,bench}.json`.

Rough cost on one RTX 3070-class GPU (not measured, order of magnitude only): MobileNetV3 ≈ 10 min
per epoch, ResNet-50 ≈ 20 min per epoch, SwinV2-B ≈ the student's epoch time without teacher
forward passes (so noticeably faster than the student's training).

## Methods paragraph (LaTeX)

```latex
\paragraph{Convolutional baselines and distillation ablation.}
To quantify the contribution of the transformer backbone and of the distillation objective
separately, three supervised classifiers were trained under the protocol of
Section~\ref{sec:protocol}. ResNet-50~\cite{he2016deep} and MobileNetV3-Large~\cite{howard2019searching},
initialised from ImageNet-1k weights, represent accuracy-oriented and edge-oriented
convolutional architectures, respectively; their final linear layers were replaced by a
four-way head over the classes \textit{fire}, \textit{smoke}, \textit{both} and \textit{neither}.
Both networks were optimised with AdamW (learning rate $3\times10^{-4}$, weight decay $0.05$)
under a one-epoch linear warm-up followed by cosine decay, the learning rate being selected
from $\{10^{-4}, 3\times10^{-4}, 10^{-3}\}$ on the validation split. In addition, the student
architecture (SwinV2-B) was trained with the student's optimiser settings but with the
cross-entropy term alone, i.e.\ without teacher supervision; the difference between this model
and the distilled student therefore isolates the effect of multi-teacher knowledge
distillation. All three models share the student's training data, augmentation, input
resolution ($192\times192$), epoch budget, checkpoint-selection rule and random seeds, and were
exported through the same ONNX--TensorRT FP16 pipeline. Region-based detectors such as
Faster R-CNN were not adopted as backbone baselines because they address a localisation task
rather than image-level classification and require non-standard TensorRT plugins for
deployment on the target platform; the detection paradigm is instead represented by the YOLO
baseline described in Section~\ref{sec:yolo-baseline}.
```
