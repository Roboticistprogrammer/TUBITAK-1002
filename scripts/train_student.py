from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms

from bootstrap import setup_path

ROOT = setup_path()

from firecls.config import DEFAULT_IMG_SIZE, get_dataset_specs
from firecls.data.dataset import ImageClassificationCSVDataset
from firecls.distillation.checkpointing import save_checkpoint
from firecls.distillation.trainer import MultiTeacherDistillationTrainer
from firecls.models.student import build_student_model
from firecls.models.teacher import load_teachers
from firecls.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a student with multi-teacher distillation.")
    parser.add_argument("--cv-teacher", type=Path, required=True)
    parser.add_argument("--rs-teacher", type=Path, required=True)
    parser.add_argument("--uav-teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/students"))
    parser.add_argument("--student-model", type=str, default="microsoft/swinv2-base-patch4-window12-192-22k")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--aggregation", type=str, default="weighted_avg")
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


def build_domain_datasets(split: str, classes, transform) -> dict[str, ImageClassificationCSVDataset]:
    specs = get_dataset_specs(ROOT)
    datasets = {}
    for key, spec in specs.items():
        index_csv = Path("data_index") / f"{spec.name.lower()}_tdml.csv"
        if not index_csv.exists():
            raise SystemExit(f"CSV not found: {index_csv}. Run prepare_tdml_classification.py first.")
        datasets[key] = ImageClassificationCSVDataset(index_csv, split, classes, transform=transform)
    return datasets


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teachers, classes, model_name = load_teachers(
        {
            "cv": args.cv_teacher,
            "rs": args.rs_teacher,
            "uav": args.uav_teacher,
        },
        device=device,
    )

    student, image_processor = build_student_model(classes, model_name=args.student_model)
    trainer = MultiTeacherDistillationTrainer(
        student_model=student,
        teacher_models=teachers,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        aggregation=args.aggregation,
        device=device,
    )

    train_tf, eval_tf = build_transforms(args.img_size, image_processor.image_mean, image_processor.image_std)
    train_sets = build_domain_datasets("train", classes, train_tf)
    val_sets = build_domain_datasets("val", classes, eval_tf)

    train_dataset = ConcatDataset(list(train_sets.values()))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loaders = {
        name: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        for name, dataset in val_sets.items()
    }

    optimizer = torch.optim.AdamW(
        trainer.student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_avg = 0.0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(train_loader, optimizer, epoch)
        val_metrics = trainer.evaluate_domains(val_loaders)
        avg_val = sum(val_metrics.values()) / max(1, len(val_metrics))

        record = {
            "epoch": epoch,
            "train_loss": metrics.loss,
            "train_hard_loss": metrics.hard_loss,
            "train_soft_loss": metrics.soft_loss,
            "train_acc": metrics.accuracy,
            "val_acc": val_metrics,
            "avg_val": avg_val,
        }
        history.append(record)
        save_json(args.output_dir / "history.json", {"records": history})

        checkpoint = {
            "epoch": epoch,
            "model": trainer.student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "classes": classes,
            "model_name": args.student_model,
            "avg_val": avg_val,
        }
        save_checkpoint(args.output_dir / "last.pt", checkpoint)
        if avg_val > best_avg:
            best_avg = avg_val
            save_checkpoint(args.output_dir / "best.pt", checkpoint)

        print(
            f"Epoch {epoch}: loss={metrics.loss:.4f} hard={metrics.hard_loss:.4f} soft={metrics.soft_loss:.4f} "
            f"train_acc={metrics.accuracy:.4f} avg_val={avg_val:.4f}"
        )


if __name__ == "__main__":
    main()
