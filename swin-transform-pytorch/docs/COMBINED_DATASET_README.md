# Combined Dataset Training Quickstart

Use this checklist whenever you want to train the Swin Transformer on the merged FASDD datasets.

## 2. Configure Datasets (config.py)

- Ensure `USE_MULTI_DATASETS = True`.
- Update `DATASET_SOURCES` paths so that each dataset entry (`FASDD_RS`, `FASDD_CD`, `FASDD_UAV`) points to the correct folder under `datasets/`.
- Control which sources participate in each split using `TRAIN_DATASETS`, `VAL_DATASETS`, and `TEST_DATASETS` lists.
- Keep `TASK` and `NUM_CLASSES` consistent across datasets (all three must share the same label space).

## 3. Running Training & Evaluation

- Training command stays the same:
  ```powershell
  python train.py
  ```
- `train.py` now loads all requested datasets, concatenates their train/val splits, and trains a single Swin model.
- At the end it automatically evaluates:
  - Combined test loader (all datasets together)
  - Individual test loaders per dataset (RS, CD, UAV) and saves the metrics in `results/test_results.txt`.

## 4. Things to Remember

- **GPU Memory**: Combined datasets increase batch count; monitor VRAM and reduce `BATCH_SIZE` or `IMG_SIZE` if you hit OOM.
- **Class Weights**: When `USE_CLASS_WEIGHTS = True`, weights are computed over the merged training set (accounts for class imbalance across sources).
- **Transforms**: All datasets share the same transform pipeline, so align preprocessing of the raw images beforehand (resolution, channel order).
- **Checkpointing**: Best model (by validation accuracy) is stored at `checkpoints/best_model.pth`; resume training by toggling `RESUME`/`RESUME_CHECKPOINT` as usual.
- **Logs**: TensorBoard logs combine statistics from every source—tag them per run to avoid confusion.

That’s it—you still use `train.py`, but now the config controls how many datasets participate in each phase.