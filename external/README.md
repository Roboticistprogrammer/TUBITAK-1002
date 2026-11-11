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
| Loader and evaluation code | `src/firecls/data/ustc.py`, `scripts/build_ustc_index.py`, `scripts/pipeline/02_dump_predictions.py` (`--dataset ustc`), `scripts/evaluate_external.py`, `tests/test_external_ustc.py` | Yes, on this branch |
| Predictions, tables, figures | `results` branch, `external/ustc_smokers/` | Yes, on `results` only |

Do not place the dataset outside `datasets/`: only that folder is ignored on all branches. Before every commit, check
that `git status` lists no `.tif` file or dataset folder.

When the evaluation is complete, merge this branch into `retrain-pipeline`, so that branch can reproduce every thesis
result.

## Commands (after the dataset is in `datasets/USTC_SmokeRS/<class>/`)

```powershell
$env:PYTHONPATH = "src"
python scripts/build_ustc_index.py --dataset-root <dir that contains datasets/>      # index + format check
python scripts/pipeline/02_dump_predictions.py --dataset ustc --name teacher_rs `
    --checkpoint outputs/teachers/FASDD_RS/best.pt --dataset-root <same dir> --limit 50   # smoke test, then drop --limit
# CNN baselines need --family cnn; engines on the Jetson use --engine instead of --checkpoint
python scripts/evaluate_external.py --preds preds --names kd_routed_seed42 kd_uniform_seed42 nokd_seed42 `
    mobilenet_v3_large_seed42 resnet50_seed42 --reference teacher_rs --out-dir results/external/ustc_smokers
```

Checkpoints are in the main checkout's `outputs/` (git-ignored), so pass the absolute path when running from a worktree.

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
- [x] (script ready: `scripts/build_ustc_index.py`; the index itself needs the dataset) Write `data_index/ustc_smokers.csv` with columns `path,label6,label_bin,split` (`split` = `test` for all images).
- [x] Add a USTC option to `02_dump_predictions.py` that reads this index, uses the same eval transform, and saves
      `logits`, `labels6`, `labels_bin` and `names`.
- [ ] (code path rehearsed on a 60-image stand-in folder, not on real USTC data) Smoke test: one model, about 50 images (`--limit 50`); if nearly every prediction is one class, check
      preprocessing before going on.

### Phase 4: Inference
- [ ] Jetson: the five TensorRT FP16 engines listed in the protocol.
- [ ] Main machine: the RS teacher (PyTorch FP32), as reference. (Also possible here: the other five as PyTorch FP32 references.)
- [ ] Each run writes `meta.json` with the commit of this branch and the engine SHA-256.

### Phase 5: Metrics and figures
- [x] (written and unit-tested; not yet run on real USTC predictions) `scripts/evaluate_external.py`: smoke recall, false-alarm rate (overall and per negative class), balanced accuracy,
      ROC-AUC, predicted-class distribution on smoke images, bootstrap CIs, McNemar with Holm correction.
- [ ] Figures: false-alarm rate per negative class and model; 4–6 example misclassifications.
- [ ] Copy predictions, tables and figures to the `results` branch under `external/ustc_smokers/`.

### Phase 6: Thesis
- [ ] Chapter 3: external-validation subsection (dataset, mapping, rule, metrics), taken from the protocol.
- [ ] Chapter 4.7: results tables and figure, without interpretation.
- [ ] Chapter 5: discussion and limitations (no fire class, 1 km resolution, different sensor).
- [ ] Appendix: full per-class tables.
- [ ] Merge this branch into `retrain-pipeline`.
