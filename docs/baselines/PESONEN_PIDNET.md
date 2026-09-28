# Baseline: SAM pseudo-masks → PIDNet-S (Pesonen et al., WACV 2025)

**Provenance: re-implemented (†).** The authors' code is hosted on GitLab, which could not be
reached when this branch was written. The pipeline below was rebuilt from the paper
(J. Pesonen et al., *Detecting Wildfires on UAVs with Real-time Segmentation Trained by Larger
Teacher Models*, WACV 2025, arXiv:2408.10843) and the official PIDNet repository. Only the PIDNet
network definition is third-party code (`third_party/pidnet/`, MIT, upstream commit `4c158cf`,
unmodified).

## 1. Method in one paragraph

A frozen foundation model (Segment Anything, ViT-H) converts every ground-truth bounding box
into a pixel mask (zero-shot, one box prompt per box). These pseudo-masks supervise a real-time
segmentation network, PIDNet-S, whose three branches (P: detail, I: context, D: boundary) are
trained with PIDNet's four-term loss. At inference only the student runs.

## 2. Reproduced versus adapted

| Aspect | Pesonen et al. | This branch | Status |
|---|---|---|---|
| Teacher | SAM ViT-H, box prompts, zero-shot | same (`sam_vit_h_4b8939.pth`, `multimask_output=False`) | reproduced |
| BoxSnake teachers | evaluated as alternatives | not implemented | out of scope |
| Student | PIDNet-S, ImageNet-pretrained | same (upstream `PIDNet_S_ImageNet.pth.tar`) | reproduced |
| Edge target | Canny on the label (PIDNet) | Canny, 6-px border suppressed, 4×4 dilation (upstream `gen_sample`) | reproduced |
| Loss | λ0=0.4 P-BCE, λ1=20 weighted boundary BCE, λ2=1 main BCE, λ3=1 BAS BCE, t=0.8 | same | reproduced |
| Optimiser | AdamW, lr 1e-3, wd 1e-2, batch 16 | same; constant lr (no schedule is stated in the paper) | reproduced / assumption |
| Augmentation | one of {crop, v-flip, rotation, perspective, erasing, grayscale, blur, invert, sharpness, colour jitter} with equal probability + 50 % h-flip | same set and rule; magnitudes are ours (§5) | reproduced / assumption |
| Classes | smoke only (binary) | **fire and smoke as two independent sigmoid channels** (a pixel may be both) | adapted |
| Input | 1080×1920 frames | **512×512 letterbox** (full frame kept, grey padding), stored in the checkpoint | adapted |
| Epochs | 50 | **20** (shared protocol; FASDD train = 61,323 images ≈ 19× their 3,252) | adapted |
| Training data | AI For Mankind + own UAV data | FASDD CV + RS + UAV `train` splits concatenated | adapted |
| Model selection | best validation checkpoint | best domain-macro **image-level val accuracy** at 0.5/0.5 (protocol); pseudo-label mIoU logged and selectable | adapted |
| Output | per-pixel smoke mask; temporal filtering over video | **image-level decision**: per-channel mean of top-K pixel probabilities, thresholds calibrated on val | adapted |

### The image-level rule (our adaptation)

The thesis compares every method as a four-class image classifier (`fire, smoke, both, neither`).
For each channel c the main-branch logits (1/8 input resolution, 64×64 at S = 512) are passed
through a sigmoid and reduced to a presence probability

  p_c = mean of the K largest pixel probabilities,  K = max(1, round(φ · H_out · W_out)),  φ = 0.001

(K = 4 at S = 512). A top-K mean is used instead of the single maximum so that one spurious pixel
cannot flip the decision, while K stays small enough that a distant, small fire still registers.
φ is a hyper-parameter stored in the checkpoint; K ≤ 3840 is asserted because TensorRT 8.x (Jetson
Nano/Xavier) limits TopK to 3840. The pair (p_fire, p_smoke) is mapped to the four scores by the
shared `PresenceToScores` head with thresholds (t_fire, t_smoke) grid-searched on the **validation**
split (`scripts/pidnet/calibrate_thresholds.py`, objective: domain-macro macro-F1). ImageNet
normalisation, PIDNet, the sigmoid, TopK and the threshold head are one module, so the ONNX/TensorRT
graph reproduces the PyTorch decision exactly (verified at export).

The paper instead filters detections temporally over video frames; FASDD consists of independent
still images, so no temporal rule is applicable.

## 3. Weights (download manually)

| File | Source | Place at |
|---|---|---|
| `sam_vit_h_4b8939.pth` (2.4 GB) | <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth> | `third_party/weights/` |
| `PIDNet_S_ImageNet.pth.tar` | <https://drive.google.com/file/d/1hIBp_8maRr60-B3PF0NVtaA6TYBvO4y-/view?usp=sharing> (fallback: the shared folder linked in `third_party/pidnet/README.md`) | `third_party/weights/` |

`third_party/weights/` and `pseudo_masks/` are git-ignored. The SAM checkpoint SHA-256 is recorded in
`pseudo_masks/manifest.json`; the ImageNet file's SHA-256 and the number of copied tensors are
recorded in every training checkpoint (`init`). Without `--pretrained` the student is randomly
initialised and a warning is printed; that setting does **not** reproduce the paper.

## 4. Commands

```bash
pip install -r requirements.txt -r requirements-baselines.txt

# 1. SAM pseudo-masks for train + val of all domains (never test); resumable, re-run to continue.
python scripts/pidnet/generate_sam_masks.py \
    --sam-checkpoint third_party/weights/sam_vit_h_4b8939.pth \
    --cv-boxes <FASDD_CV COCO .json file(s) or YOLO labels dir> \
    --rs-boxes <FASDD_RS ...> --uav-boxes <FASDD_UAV ...> \
    [--yolo-class-names fire smoke] [--limit 50]          # --limit for a dry run

# 2. Train one seed (outputs/baselines/pidnet_s/seed42/{best,last}.pt, history.json, done.json)
python scripts/pidnet/train_pidnet.py --seed 42 --amp \
    --pretrained third_party/weights/PIDNet_S_ImageNet.pth.tar
#    options: --img-size 512 --topk-fraction 0.001 --select-by {image_acc,miou}
#             --val-fraction 1.0 --resume

# 3. Calibrate thresholds on val -> outputs/baselines/pidnet_s/seed42/thresholds.json
python scripts/pidnet/calibrate_thresholds.py --checkpoint outputs/baselines/pidnet_s/seed42/best.pt

# 4-6. Shared protocol scripts (docs/BASELINES.md) with --family pidnet --name pidnet_s_seed42

# Everything, seeds 42/43/44, resumable:
CV_BOXES=... RS_BOXES=... UAV_BOXES=... scripts/pidnet/run_all.sh
```

**Cost.** Train + val comprise 102,199 images (test: 20,435, never masked). SAM ViT-H encodes each
image that has at least one box once (images without boxes are written as empty masks without
running SAM); as an unmeasured estimate, budget 0.3–0.6 s per image on a desktop GPU (about half a day),
which is why generation is resumable and done once for all seeds. Validation in every epoch covers
40,876 images at 512×512; `--val-fraction` (e.g. 0.25) reduces this if needed and must then be
reported.

`run_all.sh` skips each stage whose output is newer than its inputs, resumes interrupted training
from `last.pt`, and builds a TensorRT FP16 engine only when `tensorrt` is importable
(`BUILD_ENGINE=auto|1|0`). Engines for the Jetson must be built on the Jetson (docs/BASELINES.md).

### Files written

| Path | Content |
|---|---|
| `pseudo_masks/<domain>/<stem>.png` | uint8, original resolution, bit0 = fire, bit1 = smoke |
| `pseudo_masks/manifest.json` | per-domain counts (boxes, SAM fall-backs, box/label disagreements), SAM SHA-256, timing, `complete` flag |
| `outputs/baselines/pidnet_s/seed<k>/best.pt` | selected checkpoint (model, classes, img_size, topk_fraction, epoch, seed, history, init) |
| `…/history.json` | per epoch: train loss terms; per domain val image accuracy, macro-F1, sample-wise and pooled IoU |
| `…/thresholds.json` | calibrated (t_fire, t_smoke), objective, per-domain val metrics, checkpoint SHA-256 |
| `…/val_presence.npz` | val presence probabilities used for calibration |

`thresholds.json` stores the SHA-256 of the checkpoint it belongs to; the family loader refuses a
mismatching pair and falls back to 0.5/0.5 (with `thresholds_calibrated: false` in the metadata and
a warning) only when the file is absent.

## 5. Implementation choices not fixed by the paper

* **Augmentation magnitudes** (torchvision `transforms.v2`, applied after letterboxing):
  RandomResizedCrop scale (0.25, 1); rotation ±30°; perspective distortion 0.5; erasing
  scale (0.02, 0.33), ratio (0.3, 3.3), value 0; Gaussian blur kernel 9, σ ∈ [0.1, 3]; sharpness
  factor 2; colour jitter (0.4, 0.4, 0.4, 0.1). Geometric operations transform image and mask
  jointly (nearest-neighbour for the mask; padding fill 114/255 for the image, 0 for the mask).
  Erasing also zeroes the mask inside the erased rectangle, since the target is no longer visible.
* **SAM post-processing:** none beyond the per-class union of box masks. If SAM raises or returns
  an empty mask, the filled box is used and counted in the manifest.
* **Missing box entries:** an image without any box entry receives an all-zero mask; disagreements
  between the CSV label and the box-implied label are counted in the manifest, not corrected
  (the CSV label is the ground truth shared by all methods).
* **Validation subsampling:** `--val-fraction f < 1` scores a fixed, seeded subset of each domain's
  val split every epoch; the default 1.0 uses the full split. Threshold calibration always uses the
  full val split.
* **Pseudo-label mIoU** (`history.json`): sample-wise Jaccard index averaged over the images whose
  pseudo-mask contains the class, following the paper's definition (mean of sample-wise Jaccard,
  binary per class); dataset-level (pooled) IoU is logged alongside. It measures agreement with the
  SAM teacher on val, not with manual masks.
* **Determinism:** `firecls.utils.set_seed`, and each epoch reseeds shuffling and augmentation from
  `seed·1000 + epoch`, so a resumed run follows the same sample order as an uninterrupted one.

## 6. Reported numbers (context only — **not comparable**)

| Quantity (as reported by Pesonen et al.) | Value | Why not comparable |
|---|---|---|
| Smoke mIoU, manually annotated test set (80 images), PIDNet-S + SAM teacher | 63.3 % ‡ | different data, pixel-level metric, smoke only |
| Throughput on a UAV-carried Jetson Orin NX (abstract) | ~25 fps ‡ | 1080×1920 input, TensorRT; our input and devices differ |

‡ reported, copied from the paper. The thesis table reports this baseline only through the shared
image-level protocol (test split, per-domain accuracy / balanced accuracy / macro-F1, latency at
the method's own 512×512 input).

## 7. Threats to validity specific to this baseline

* Details absent from the paper (augmentation magnitudes, learning-rate schedule, SAM mask
  post-processing) were chosen as in §5 and may differ from the authors' code.
* The student is trained for 20 instead of 50 epochs, at 512×512 instead of full HD, and on two
  classes instead of one; the result characterises the *training strategy* under the thesis
  protocol, not the published model.
* Image-level decisions depend on the top-K fraction φ, fixed a priori at 0.001 and not tuned on
  test; the two thresholds are the only quantities fitted, and only on val.
* Latency is measured at 512×512 (the student runs at 192×192); this is stated alongside the table.

## 8. Methods paragraph (LaTeX)

```latex
\paragraph{Box-supervised segmentation baseline.}
As a representative of weakly supervised, segmentation-based wildfire detection, we
re-implemented the teacher--student approach of Pesonen et al.~\cite{pesonen2025wildfire}, for
which no reachable reference implementation was available at the time of writing. A frozen
Segment Anything model with a ViT-H image encoder~\cite{kirillov2023sam} was prompted zero-shot
with each ground-truth bounding box of the FASDD training and validation images
(single-mask output), and the resulting masks were merged per class into pixel-level pseudo-labels;
test images were never processed. These pseudo-labels supervised PIDNet-S~\cite{xu2023pidnet},
initialised from the authors' ImageNet weights, using the original four-term objective
($\lambda_0 = 0.4$ for the auxiliary detail branch, $\lambda_1 = 20$ for the class-balanced boundary
term on Canny-derived edges, $\lambda_2 = 1$ for the main output and $\lambda_3 = 1$ for the
boundary-awareness term at threshold $t = 0.8$). Optimisation followed the published recipe
(AdamW, learning rate $10^{-3}$, weight decay $10^{-2}$, batch size 16) together with a single
uniformly sampled augmentation per image and an independent horizontal flip. Three adaptations
were required to place the method within the present comparison. First, fire and smoke were
modelled as two independent sigmoid channels, whereas the original work segments smoke alone.
Second, images were letterboxed to $512 \times 512$ pixels and training was limited to the
20-epoch budget shared by all methods, since the FASDD training split is roughly nineteen times larger than the
original training set. Third, because the thesis evaluates image-level decisions on still images
rather than temporally filtered video, the presence of each class was summarised as the mean of
the $K = \max(1, \mathrm{round}(0.001\,HW))$ highest pixel probabilities of the $H \times W$ output
map, and the two decision thresholds were calibrated on the validation split only. The checkpoint
with the highest domain-averaged validation accuracy was retained, and the complete inference
chain, including normalisation, top-$K$ pooling and thresholding, was exported as a single ONNX
graph and compiled with TensorRT under the same protocol as all other models. Reported figures
for the original system (63.3\,\% smoke mIoU; approximately 25 frames per second on a Jetson Orin NX)
refer to a different dataset, task and input resolution and are therefore quoted for context only.
```

Suggested BibTeX keys:

```bibtex
@inproceedings{pesonen2025wildfire,
  author    = {Pesonen, Julius and Hakala, Teemu and Karjalainen, V{\"a}in{\"o} and Koivum{\"a}ki, Niko and
               Markelin, Lauri and Raita-Hakola, Anna-Maria and Suomalainen, Juha and
               P{\"o}l{\"o}nen, Ilkka and Honkavaara, Eija},
  title     = {Detecting Wildfires on {UAVs} with Real-Time Segmentation Trained by Larger Teacher Models},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2025},
  note      = {arXiv:2408.10843}
}
@inproceedings{kirillov2023sam,
  author    = {Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and
               Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and
               Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  title     = {Segment Anything},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023}
}
@inproceedings{xu2023pidnet,
  author    = {Xu, Jiacong and Xiong, Zixiang and Bhattacharyya, Shankar P.},
  title     = {{PIDNet}: A Real-Time Semantic Segmentation Network Inspired by {PID} Controllers},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}
```
