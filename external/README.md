# External validation: USTC_SmokeRS

Branch `external/ustc-smokers`, created from `retrain-pipeline` at `b6b9188`. It holds the **code** for evaluating the
deployed models on the USTC_SmokeRS satellite smoke dataset. The rules of the evaluation are fixed in
[`USTC_PROTOCOL.md`](USTC_PROTOCOL.md) and must be committed before any model is run.

## Where things go

| What | Where | Committed? |
|---|---|---|
| Dataset images | `datasets/USTC_SmokeRS/` | No: ignored on every branch (`datasets/`, `*.tif`) |
| Protocol, this plan | `external/` | Yes, on this branch |
| Image index | `data_index/ustc_smokers.csv` (+ `.meta.json`) | Yes, on this branch |
| Loader and evaluation code | `scripts/pipeline/02_dump_predictions.py` (USTC option), `scripts/evaluate_external.py` | Yes, on this branch |
| Predictions, tables, figures | `results` branch, `external/ustc_smokers/` | Yes, on `results` only |

Do not place the dataset outside `datasets/`: only that folder is ignored on all branches. Before every commit, check
that `git status` lists no `.tif` file or dataset folder.

When the evaluation is complete, merge this branch into `retrain-pipeline`, so that branch can reproduce every thesis
result.

## Plan

### Phase 0: Fix the rules (before running anything)
- [ ] Review `USTC_PROTOCOL.md` (models, label mapping, prediction rule, metrics) and commit it.

### Phase 1: Download
- [ ] Download from the Baidu Pan link given in the SmokeNet paper (*Remote Sensing* 11(14):1702; password in the paper).
      Baidu Pan usually needs an account, and large files its desktop client.
- [ ] If that fails, try the USTC lab page (complex.ustc.edu.cn), then mirrors; accept a mirror only if its per-class
      counts match the paper.
- [ ] Extract to `datasets/USTC_SmokeRS/`; fill in the download record in the protocol (source, date, SHA-256).
- [ ] Fallback: if the data cannot be obtained within about 3 days, switch to the Sen2Fire test split (Zenodo) and
      inform the advisor.

### Phase 2: Inspect
- [ ] Count images per class (expected total 6,225).
- [ ] Check bit depth, channels and channel order; confirm `PIL.Image.open(...).convert("RGB")` gives correct images.
- [ ] Compare 10 USTC and 10 FASDD-RS images loaded the same way.
- [ ] Fill in the image format check in the protocol; fix the loader first if needed.

### Phase 3: Index and loader
- [ ] Write `data_index/ustc_smokers.csv` with columns `path,label6,label_bin,split` (`split` = `test` for all images).
- [ ] Add a USTC option to `02_dump_predictions.py` that reads this index, uses the same eval transform, and saves
      `logits`, `labels6`, `labels_bin` and `names`.
- [ ] Smoke test: one model, about 50 images (`--limit 50`); if nearly every prediction is one class, check
      preprocessing before going on.

### Phase 4: Inference
- [ ] Jetson: the five TensorRT FP16 engines listed in the protocol.
- [ ] Main machine: the RS teacher (PyTorch FP32), as reference.
- [ ] Each run writes `meta.json` with the commit of this branch and the engine SHA-256.

### Phase 5: Metrics and figures
- [ ] `scripts/evaluate_external.py`: smoke recall, false-alarm rate (overall and per negative class), balanced accuracy,
      ROC-AUC, predicted-class distribution on smoke images, bootstrap CIs, McNemar with Holm correction.
- [ ] Figures: false-alarm rate per negative class and model; 4–6 example misclassifications.
- [ ] Copy predictions, tables and figures to the `results` branch under `external/ustc_smokers/`.

### Phase 6: Thesis
- [ ] Chapter 3: external-validation subsection (dataset, mapping, rule, metrics), taken from the protocol.
- [ ] Chapter 4.7: results tables and figure, without interpretation.
- [ ] Chapter 5: discussion and limitations (no fire class, 1 km resolution, different sensor).
- [ ] Appendix: full per-class tables.
- [ ] Merge this branch into `retrain-pipeline`.
