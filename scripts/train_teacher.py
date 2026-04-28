from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from bootstrap import setup_path

ROOT = setup_path()

from firecls.config import DEFAULT_CLASSES, DEFAULT_IMG_SIZE, DEFAULT_MODEL_NAME, get_dataset_specs
from firecls.data.dataset import ImageClassificationCSVDataset
from firecls.data.tdml import create_index_csv
from firecls.models.swinv2 import build_swinv2_classifier
from firecls.utils import accuracy_from_logits, save_json, set_seed, to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SwinV2-L teacher on one FASDD domain.")
    parser.add_argument("--dataset", choices=["cv", "rs", "uav"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/teachers"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--both-as", choices=["both", "fire", "smoke"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def build_transforms(img_size: int, mean, std):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, eval_tf


def evaluate(model, loader, device):
    model.eval()
    total_acc = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = to_device((images, labels), device)
            outputs = model(images)
            acc = accuracy_from_logits(outputs.logits, labels)
            total_acc += acc * labels.size(0)
            total += labels.size(0)
    return total_acc / max(1, total)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    specs = get_dataset_specs(ROOT)
    spec = specs[args.dataset]
    output_dir = args.output_dir / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)

    index_csv = Path("data_index") / f"{spec.name.lower()}_tdml.csv"
    if not index_csv.exists():
        counts = create_index_csv(spec.tdml_json, spec.images_root, index_csv, both_as=args.both_as, limit=args.limit)
        save_json(index_csv.with_suffix(".meta.json"), {
            "tdml_json": str(spec.tdml_json),
            "images_root": str(spec.images_root),
            "counts": counts,
            "classes": DEFAULT_CLASSES,
        })

    model, image_processor = build_swinv2_classifier(
        num_labels=len(DEFAULT_CLASSES),
        label2id={name: i for i, name in enumerate(DEFAULT_CLASSES)},
        id2label={i: name for i, name in enumerate(DEFAULT_CLASSES)},
        model_name=DEFAULT_MODEL_NAME,
    )

    train_tf, eval_tf = build_transforms(args.img_size, image_processor.image_mean, image_processor.image_std)

    train_ds = ImageClassificationCSVDataset(index_csv, "train", DEFAULT_CLASSES, transform=train_tf)
    val_ds = ImageClassificationCSVDataset(index_csv, "val", DEFAULT_CLASSES, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    best_val = 0.0

    if args.resume and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val = checkpoint.get("best_val", 0.0)

    history = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels in loop:
            images, labels = to_device((images, labels), device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc = accuracy_from_logits(outputs.logits, labels)
            running_loss += loss.item() * labels.size(0)
            running_acc += acc * labels.size(0)
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=acc)

        train_loss = running_loss / max(1, total)
        train_acc = running_acc / max(1, total)
        val_acc = evaluate(model, val_loader, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        history.append(record)

        save_json(output_dir / "history.json", {"records": history})

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val": best_val,
            "classes": DEFAULT_CLASSES,
            "model_name": DEFAULT_MODEL_NAME,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(checkpoint, output_dir / "best.pt")

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()
