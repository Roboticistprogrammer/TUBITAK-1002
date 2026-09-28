"""Supervised image classifiers under the shared protocol (no distillation).

Architectures
-------------
* ``resnet50``, ``mobilenet_v3_large`` -- torchvision, ImageNet-1k initialisation.
* ``swinv2_base_nokd`` -- the student's exact architecture and optimiser, trained with
  cross-entropy only. It isolates the contribution of multi-teacher distillation: any gap
  between this row and the student is attributable to the KD objective, not the backbone.

Everything that defines the comparison (data, splits, augmentation, input geometry, epoch
budget, selection rule, seeds) comes from ``firecls.baselines.protocol``. Only the optimiser
recipe is architecture-specific, and each recipe is recorded in the checkpoint.

    python scripts/cnn/train_classifier.py --arch resnet50 --seed 42 --amp
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import setup_path  # noqa: E402

ROOT = setup_path()

from firecls.baselines import protocol  # noqa: E402
from firecls.baselines.data import combined_train_loader, domain_loaders  # noqa: E402
from firecls.baselines.families.cnn import ARCHITECTURES, build_cnn  # noqa: E402
from firecls.baselines.preprocessing import IMAGENET_MEAN, IMAGENET_STD, build_transform, resize_center_crop_spec  # noqa: E402
from firecls.utils import save_json, set_seed  # noqa: E402

STUDENT_BACKBONE = "microsoft/swinv2-base-patch4-window12-192-22k"

# Optimiser recipes. CNNs use a standard ImageNet fine-tuning recipe (AdamW + cosine with
# warm-up); forcing the transformer student's lr of 3e-5 onto a CNN would handicap it. The
# no-KD SwinV2 copies train_student.py exactly (AdamW 3e-5, wd 0.01, constant lr, clip 1.0).
RECIPES = {
    "cnn": {"optimizer": "AdamW", "lr": 3e-4, "weight_decay": 0.05, "schedule": "cosine", "warmup_epochs": 1, "grad_clip": 1.0},
    "swinv2_base_nokd": {"optimizer": "AdamW", "lr": 3e-5, "weight_decay": 0.01, "schedule": "constant", "warmup_epochs": 0, "grad_clip": 1.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES) + ["swinv2_base_nokd"], required=True)
    parser.add_argument("--seed", type=int, default=protocol.SEEDS[0])
    parser.add_argument("--epochs", type=int, default=protocol.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=protocol.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=None, help="override the recipe lr (for the tuning sweep)")
    parser.add_argument("--img-size", type=int, default=protocol.CLASSIFIER_IMG_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=ROOT, help="directory containing datasets/")
    parser.add_argument("--index-root", type=Path, default=ROOT, help="directory containing data_index/")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="continue from last.pt in the output dir")
    parser.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
    return parser.parse_args()


def build_model(arch: str, pretrained: bool):
    if arch == "swinv2_base_nokd":
        from firecls.models.swinv2 import build_swinv2_classifier

        model, processor = build_swinv2_classifier(
            num_labels=len(protocol.CLASSES),
            label2id={c: i for i, c in enumerate(protocol.CLASSES)},
            id2label={i: c for i, c in enumerate(protocol.CLASSES)},
            model_name=STUDENT_BACKBONE,
            pretrained=pretrained,
        )
        return model, list(processor.image_mean), list(processor.image_std)
    return build_cnn(arch, len(protocol.CLASSES), pretrained), IMAGENET_MEAN, IMAGENET_STD


def logits_of(outputs) -> torch.Tensor:
    return getattr(outputs, "logits", outputs)


def train_transform(img_size: int, mean, std):
    """Identical to train_student.py so augmentation is not a confounder."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def lr_lambda(recipe: dict, steps_per_epoch: int, epochs: int):
    warmup = recipe["warmup_epochs"] * steps_per_epoch
    total = max(1, epochs * steps_per_epoch)

    def factor(step: int) -> float:
        if recipe["schedule"] == "constant":
            return 1.0
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return factor


@torch.no_grad()
def evaluate_domains(model, loaders, device) -> dict[str, float]:
    model.eval()
    accuracies = {}
    for domain, loader in loaders.items():
        correct = total = 0
        for images, labels in loader:
            predictions = logits_of(model(images.to(device, non_blocking=True))).argmax(1).cpu()
            correct += int((predictions == labels).sum())
            total += labels.numel()
        accuracies[domain] = correct / max(1, total)
    return accuracies


def checkpoint_payload(args, model, recipe, epoch, avg_val, history) -> dict:
    payload = {
        "model": model.state_dict(),
        "classes": protocol.CLASSES,
        "epoch": epoch,
        "avg_val": avg_val,
        "seed": args.seed,
        "img_size": args.img_size,
        "recipe": recipe,
        "protocol": protocol.protocol_dict(),
        "history": history,
    }
    if args.arch == "swinv2_base_nokd":
        payload["model_name"] = STUDENT_BACKBONE  # loadable by the existing "swinv2" family
    else:
        payload["arch"] = args.arch
    return payload


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recipe = dict(RECIPES["swinv2_base_nokd" if args.arch == "swinv2_base_nokd" else "cnn"])
    if args.lr is not None:
        recipe["lr"] = args.lr
    output_dir = args.output_dir or ROOT / "outputs" / "baselines" / args.arch / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, mean, std = build_model(args.arch, pretrained=not args.no_pretrained)
    model.to(device)
    eval_spec = resize_center_crop_spec(args.img_size, args.img_size + 32, mean, std)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = combined_train_loader(
        train_transform(args.img_size, mean, std), args.batch_size, args.num_workers, args.dataset_root,
        seed_generator=generator, index_root=args.index_root,
    )
    val_loaders = domain_loaders(
        protocol.SELECTION_SPLIT, build_transform(eval_spec), args.batch_size, args.num_workers,
        args.dataset_root, index_root=args.index_root,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(recipe, len(train_loader), args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss()

    start_epoch, best_avg, history = 1, -1.0, []
    last_path = output_dir / "last.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_avg, history = state["epoch"] + 1, state["best_avg"], state["history"]
        print(f"Resumed from epoch {state['epoch']} (best avg val {best_avg:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        started, running_loss, correct, seen = time.time(), 0.0, 0, 0
        progress = tqdm(train_loader, desc=f"{args.arch} seed{args.seed} epoch {epoch}/{args.epochs}")
        for images, labels in progress:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=args.amp and device.type == "cuda"):
                logits = logits_of(model(images))
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item() * labels.size(0)
            correct += int((logits.argmax(1) == labels).sum())
            seen += labels.size(0)
            progress.set_postfix(loss=running_loss / seen, acc=correct / seen)

        val = evaluate_domains(model, val_loaders, device)
        avg_val = float(np.mean(list(val.values())))
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "train_acc": correct / max(1, seen),
            "val_acc": val,
            "avg_val": avg_val,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        history.append(record)
        run_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        save_json(output_dir / "history.json", {"records": history, "recipe": recipe, "args": run_args})

        if avg_val > best_avg:
            best_avg = avg_val
            torch.save(checkpoint_payload(args, model, recipe, epoch, avg_val, history), output_dir / "best.pt")
        last = checkpoint_payload(args, model, recipe, epoch, avg_val, history)
        last.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), best_avg=best_avg)
        torch.save(last, last_path)
        print(f"epoch {epoch}: loss={record['train_loss']:.4f} val={val} avg={avg_val:.4f} best={best_avg:.4f}")

    print(f"Best checkpoint: {output_dir / 'best.pt'} (avg val acc {best_avg:.4f})")


if __name__ == "__main__":
    main()
