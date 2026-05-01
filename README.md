# Fire Classification with SwinV2-L Teachers

This workspace trains **SwinV2-L (patch4-window12-192-22k)** teachers on each FASDD domain (CV/UAV/RS) using TDML annotations converted into image-level labels.

## Datasets

FASDD datasets are located under `datasets/`:

- `datasets/FASDD_CV/FASDD_CV/`
- `datasets/FASDD_RS/`
- `datasets/FASDD_UAV/`

TDML annotations are **object detection** style, so we first convert them into **image-level labels**.
By default we use 4 classes: `fire`, `smoke`, `both`, `neither`.

## Setup

Install the dependencies directly from `requirements.txt` — no `pyproject.toml` is required:

```bash
uv pip install -r requirements.txt
```

If you prefer `pip`, use `pip install -r requirements.txt`.

If you need a CUDA-specific PyTorch build, install the matching `torch`/`torchvision` wheel from the
[official PyTorch selector](https://pytorch.org/get-started/locally/) first, then run the same requirements install.

## Prepare TDML → CSV

```bash
python scripts/prepare_tdml_classification.py --dataset cv
python scripts/prepare_tdml_classification.py --dataset rs
python scripts/prepare_tdml_classification.py --dataset uav
```

This creates CSVs under `data_index/`.

## Train (Teacher)

```bash
python scripts/train_teacher.py --dataset cv --epochs 20 --batch-size 8 --amp
python scripts/train_teacher.py --dataset rs --epochs 20 --batch-size 8 --amp
python scripts/train_teacher.py --dataset uav --epochs 20 --batch-size 8 --amp
```

Checkpoints are saved to `outputs/teachers/<DATASET>/best.pt`.

## Evaluate

```bash
python scripts/evaluate_teacher.py --dataset cv --checkpoint outputs/teachers/FASDD_CV/best.pt --split test
```

## Predict (Single Image)

```bash
python scripts/predict.py --checkpoint outputs/teachers/FASDD_CV/best.pt --image datasets/FASDD_CV/FASDD_CV/images/your.jpg
```

## Train (Student Distillation)

```bash
python scripts/train_student.py \
	--cv-teacher outputs/teachers/FASDD_CV/best.pt \
	--rs-teacher outputs/teachers/FASDD_RS/best.pt \
	--uav-teacher outputs/teachers/FASDD_UAV/best.pt \
	--epochs 20 \
	--batch-size 8
```

## Check Image Sizes

```bash
python scripts/check_image_sizes.py --root datasets/FASDD_UAV/images --limit 1000
```

## Predict (Single Image with Student Model)

```bash
python scripts/predict.py --checkpoint outputs/students/best.pt --image datasets/FASDD_CV/FASDD_CV/images/bothFireAndSmoke_CV000004.jpg
```

## Notes

- TDML uses object detection labels; `prepare_tdml_classification.py` converts them to image-level classes.
- If an image contains both fire and smoke, the label is `both` by default. You can change with `--both-as fire` or `--both-as smoke`.