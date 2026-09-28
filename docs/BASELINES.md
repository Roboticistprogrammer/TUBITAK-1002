# Baseline Comparison Protocol

This document fixes how every baseline in the Results chapter is trained, exported, evaluated
and timed. The shared code lives in `src/firecls/baselines/` on the `baselines/common` branch;
each baseline lives on its own branch that starts from it.

| Branch | Method | Literature link | Provenance |
|---|---|---|---|
| `baseline/cnn-classifiers` | ResNet-50, MobileNetV3-Large, SwinV2-B without KD | He et al. 2016; Howard et al. 2019; ablation of the proposed KD | official (torchvision / Hugging Face) |
| `baseline/yolo-vazquez` | YOLOv5n / YOLOv8n / YOLO11n detectors, COCO-pretrained vs. scratch | Vazquez et al. 2025 | official library, protocol adapted (†) |
| `baseline/pesonen-pidnet` | SAM box-prompted pseudo-masks → PIDNet-S student | Pesonen et al., WACV 2025 | reimplemented (†) |

## 1. Task definition

The proposed model is an **image-level classifier** with four ordered classes
`fire, smoke, both, neither`, derived from FASDD TDML annotations
(`scripts/prepare_tdml_classification.py`). Every baseline is reduced to the same contract:

```
preprocessed image batch [B, 3, H, W]  ->  scores [B, 4]  ->  arg-max = predicted class
```

* **Classifiers** output these scores directly.
* **Detectors and segmenters** produce per-class *presence* probabilities
  (`p_fire`, `p_smoke`). `firecls.baselines.scores` converts them to the 4-way space with two
  thresholds calibrated on the **validation** split (never on test). The conversion is exact:
  the arg-max of the scores equals the thresholded decision. The thresholds are baked into
  the exported ONNX graph, so the TensorRT engine reproduces the PyTorch decision.

## 2. Fixed experimental conditions

Defined once in `src/firecls/baselines/protocol.py`:

| Item | Value |
|---|---|
| Training data | CV + RS + UAV `train` splits concatenated (as for the student) |
| Model selection | best equal-weight mean `val` accuracy over the three domains |
| Evaluation | `test` split, each domain separately; equal-weight domain macro average |
| Metrics | accuracy, balanced accuracy, macro-F1 over classes present, per-class P/R/F1, confusion matrix |
| Budget | 20 epochs, batch 16 (per-method optimiser settings documented on each branch) |
| Seeds | 42, 43, 44 → reported as mean ± sample std |
| Initialisation | ImageNet (classifiers), COCO or scratch (YOLO, both reported) |
| Classifier input | resize 224 → centre-crop 192 (student geometry) |

Rationale for per-method optimisers: forcing the student's AdamW 3e-5 onto a CNN trained from
ImageNet weights would handicap it. Each branch uses its authors' or library's recommended
recipe and states it; the budget (epochs, data, selection rule, seeds) is identical.

## 3. Pipeline for every baseline

```bash
# 1. train (branch-specific script), writes outputs/baselines/<method>/seed<k>/best.pt
# 2. export ONNX with verification + manifest
python scripts/export_baseline.py --family <family> --checkpoint <best.pt> \
    --output artifacts/baselines/<method>_seed<k>.onnx
# 3. TensorRT FP16 on the target device (same script as the student)
python scripts/build_engine.py --onnx artifacts/baselines/<method>_seed<k>.onnx \
    --output artifacts/baselines/<method>_seed<k>_fp16.engine --precision fp16
# 4. evaluate PT / ONNX / TensorRT on identical ordered samples
python scripts/evaluate_baseline.py --family <family> --name <method>_seed<k> \
    --checkpoint <best.pt> --onnx <...>.onnx --engine <...>.engine \
    --output results/baselines/<method>_seed<k>_eval.json
# 5. latency / throughput / power
python scripts/benchmark_baseline.py --family <family> --name <method>_seed<k> \
    --checkpoint <best.pt> --onnx <...>.onnx --engine <...>.engine \
    --output results/baselines/<method>_seed<k>_bench.json
# 6. one table for the thesis
python scripts/collect_results_table.py \
    --eval results/baselines/*_eval.json results/classification_evaluation.json \
    --bench results/baselines/*_bench.json results/deployment_benchmark.json \
    --accuracy-runtime engine --latency-runtime engine --batch 1 \
    --output results/baselines/comparison
```

The distilled student can be run through steps 4–5 as well with `--family swinv2`, so the
student and the baselines are scored by literally the same code.

### Jetson

Engines are not portable: copy the verified `.onnx` to the Jetson and build there. When the
JetPack's Python cannot run this repository (JetPack 4 on the original Nano ships Python 3.6),
use the version-agnostic path:

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
scripts/jetson_trtexec_benchmark.sh artifacts/baselines/resnet50_seed42.onnx 192 1
python scripts/summarize_jetson_run.py results/jetson/resnet50_seed42_b1 --name resnet50_seed42 --batch 1
```

Report the JetPack/TensorRT version, power mode and that `jetson_clocks` was applied.

## 4. Threats to validity to state in the thesis

* **Task adaptation.** Vazquez et al. and Pesonen et al. are detection/segmentation methods;
  they are evaluated here through the image-level decision rule of §1. This is stated as an
  adaptation, not a reproduction of their reported numbers.
* **Pre-training leakage.** Vazquez et al. publish FASDD-pretrained YOLO weights. They are *not*
  used: they may have seen FASDD test images. All YOLO runs start from COCO or from scratch.
* **External data.** BoWFire is part of FASDD_CV's sources; do not use it as an external test set.
* **Input geometry.** YOLO keeps its native 640 px letterbox input because down-scaling to 192
  would cripple a detector; latency is therefore reported at each method's native input size.
