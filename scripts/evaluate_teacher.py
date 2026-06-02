from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from bootstrap import setup_path

ROOT = setup_path()

from firecls.config import DEFAULT_CLASSES, DEFAULT_IMG_SIZE, DEFAULT_MODEL_NAME, get_dataset_specs
from firecls.data.dataset import ImageClassificationCSVDataset
from firecls.models.swinv2 import build_swinv2_classifier
from firecls.utils import to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SwinV2 teacher checkpoint.")
    parser.add_argument("--dataset", choices=["cv", "rs", "uav"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    return parser.parse_args()


def build_eval_transform(img_size: int, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

def main() -> None:
    args = parse_args()
    specs = get_dataset_specs(ROOT)
    spec = specs[args.dataset]

    index_csv = Path("data_index") / f"{spec.name.lower()}_tdml.csv"
    if not index_csv.exists():
        raise SystemExit(f"CSV not found: {index_csv}. Run prepare_tdml_classification.py first.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    classes = checkpoint.get("classes", DEFAULT_CLASSES)
    model_name = checkpoint.get("model_name", DEFAULT_MODEL_NAME)
    model, image_processor = build_swinv2_classifier(
        num_labels=len(classes),
        label2id={name: i for i, name in enumerate(classes)},
        id2label={i: name for i, name in enumerate(classes)},
        model_name=model_name,
    )

    model.load_state_dict(checkpoint["model"])

    eval_tf = build_eval_transform(args.img_size, image_processor.image_mean, image_processor.image_std)
    dataset = ImageClassificationCSVDataset(index_csv, args.split, classes, transform=eval_tf)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = to_device((images, labels), device)
            logits = model(images).logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=classes))


if __name__ == "__main__":
    main()
