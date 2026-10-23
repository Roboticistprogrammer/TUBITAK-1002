"""Fine-tune one existing model with the Industrial Smoke Plumes data (external/PLUME_PROTOCOL.md, Section 7).

    # 1. RS teacher first (FASDD-RS train + plume train)
    python scripts/finetune_rs_plumes.py --kind teacher --checkpoint outputs/teachers/FASDD_RS/best.pt \
        --output-dir outputs/plume_finetune/teacher_rs/seed42 --dataset-root <dir with datasets/> --seed 42
    # 2. KD students use the fine-tuned RS teacher
    python scripts/finetune_rs_plumes.py --kind kd_routed --checkpoint outputs/retrain/kd_routed/seed42/best.pt \
        --cv-teacher outputs/teachers/FASDD_CV/best.pt --rs-teacher outputs/plume_finetune/teacher_rs/seed42/best.pt \
        --uav-teacher outputs/teachers/FASDD_UAV/best.pt --output-dir outputs/plume_finetune/kd_routed/seed42 ...
    # 3. no-KD student, MobileNetV3-Large, ResNet-50: --kind nokd | mobilenet_v3_large | resnet50
    # Control (same fine-tuning without plume images): add --no-plumes

Every model keeps its original optimizer, augmentation, batch size and AMP setting (see firecls.finetune.recipes); 5 epochs at
one tenth of the original learning rate; learning-rate schedules are restarted. One epoch is one pass over the model's full
fine-tuning data. Normal sampling, unless `--balanced-rs-sampler` (the pre-registered fallback). Plume images are tagged as the
RS domain, so routed distillation takes their soft target from the (fine-tuned) RS teacher.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from bootstrap import setup_path

ROOT = setup_path()

from firecls.baselines.preprocessing import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from firecls.baselines.protocol import CLASSES  # noqa: E402
from firecls.config import DEFAULT_IMG_SIZE  # noqa: E402
from firecls.data.plumes import PLUME_INDEX  # noqa: E402
from firecls.deployment.artifacts import git_commit, sha256_file  # noqa: E402
from firecls.finetune.data import (  # noqa: E402
    build_train_dataset, build_val_datasets, rs_class_balanced_sampler, source_balanced_batch_sampler)
from firecls.finetune.recipes import FINETUNE_EPOCHS, KINDS, LR_FACTOR, finetune_recipe  # noqa: E402
from firecls.finetune.runner import make_optimizer, run_finetune, train_supervised_epoch  # noqa: E402
from firecls.utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=sorted(KINDS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="the model's current (before) best.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT, help="directory containing datasets/")
    parser.add_argument("--index-root", type=Path, default=ROOT, help="directory containing data_index/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=FINETUNE_EPOCHS)
    parser.add_argument("--lr-factor", type=float, default=LR_FACTOR, help="fraction of the original learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="override the original batch size (state it if you do)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--no-plumes", action="store_true", help="control: the same fine-tuning without plume train images")
    parser.add_argument("--balanced-rs-sampler", action="store_true", help="pre-registered fallback sampler for the RS portion")
    parser.add_argument("--source-balanced-batches", action="store_true",
                        help="RS-teacher fallback: every batch 50%% FASDD-RS train, 50%% plume train")
    parser.add_argument("--cv-teacher", type=Path)
    parser.add_argument("--rs-teacher", type=Path, help="the FINE-TUNED RS teacher for the KD students")
    parser.add_argument("--uav-teacher", type=Path)
    parser.add_argument("--limit-train", type=int, default=None, help="random subset of the training images (smoke tests only)")
    parser.add_argument("--limit-val", type=int, default=None, help="random subset of each validation set (smoke tests only)")
    return parser.parse_args()


def build_model(kind: str, checkpoint: Path):
    """Returns (model, mean, std) from the model's own checkpoint format, as the original evaluation loaders do."""
    if KINDS[kind][1]:  # SwinV2 checkpoints: teacher, KD students, no-KD student
        from firecls.deployment.model import load_checkpoint_model

        model, processor, metadata = load_checkpoint_model(checkpoint, "cpu")
        if list(metadata["classes"]) != list(CLASSES):
            raise ValueError(f"class order {metadata['classes']} != {CLASSES}")
        return model, list(processor.image_mean), list(processor.image_std)
    from firecls.baselines.registry import load_baseline

    loaded = load_baseline("cnn", checkpoint, "cpu")
    if loaded.metadata["arch"] != kind:
        raise ValueError(f"{checkpoint} holds a {loaded.metadata['arch']}, not a {kind}")
    return loaded.model, IMAGENET_MEAN, IMAGENET_STD


def dirty_files() -> list[str] | None:
    try:
        return subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recipe_key, _, all_domains, loss = KINDS[args.kind]
    recipe = finetune_recipe(args.kind, args.lr_factor, args.batch_size)
    domains = ["cv", "rs", "uav"] if all_domains else ["rs"]
    if loss == "kd" and not (args.cv_teacher and args.rs_teacher and args.uav_teacher):
        raise SystemExit("KD kinds need --cv-teacher, --rs-teacher (the fine-tuned one) and --uav-teacher")

    base = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, mean, std = build_model(args.kind, args.checkpoint)
    model.to(device)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    eval_tf = transforms.Compose([
        transforms.Resize(args.img_size + 32), transforms.CenterCrop(args.img_size),
        transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    tag = args.kind == "kd_routed"
    train_ds, sizes = build_train_dataset(domains, CLASSES, train_tf, args.dataset_root, args.index_root,
                                          include_plumes=not args.no_plumes, tag_domains=tag)
    sampler, batch_sampler, part_names = None, None, list(sizes)
    if args.balanced_rs_sampler:
        sampler = rs_class_balanced_sampler(train_ds, part_names, args.seed)
    if args.source_balanced_batches:
        if args.kind != "teacher" or args.no_plumes or args.balanced_rs_sampler:
            raise SystemExit("--source-balanced-batches is the RS-teacher fallback: --kind teacher, with plumes, "
                             "and not together with --balanced-rs-sampler")
        batch_sampler = source_balanced_batch_sampler(train_ds, part_names, [["fasdd_rs"], ["plume"]],
                                                      recipe.batch_size, args.seed)
    if args.limit_train:
        rng = np.random.default_rng(args.seed)
        keep = sorted(rng.choice(len(train_ds), min(args.limit_train, len(train_ds)), replace=False).tolist())
        train_ds = Subset(train_ds, keep)
        sampler = batch_sampler = None  # smoke tests only
    generator = torch.Generator().manual_seed(args.seed)
    if batch_sampler is not None:
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers,
                                  pin_memory=device.type == "cuda")
    else:
        train_loader = DataLoader(train_ds, batch_size=recipe.batch_size, shuffle=sampler is None, sampler=sampler,
                                  generator=generator, num_workers=args.num_workers, drop_last=recipe.drop_last,
                                  pin_memory=device.type == "cuda")
    val_sets = build_val_datasets(domains, CLASSES, eval_tf, args.dataset_root, args.index_root)
    if args.limit_val:
        rng = np.random.default_rng(args.seed)
        val_sets = {k: Subset(v, sorted(rng.choice(len(v), min(args.limit_val, len(v)), replace=False).tolist()))
                    for k, v in val_sets.items()}
    val_loaders = {k: DataLoader(v, batch_size=32, shuffle=False, num_workers=args.num_workers) for k, v in val_sets.items()}

    if loss == "kd":
        from firecls.distillation.trainer import MultiTeacherDistillationTrainer
        from firecls.models.teacher import load_teachers

        teachers, _, _ = load_teachers({"cv": args.cv_teacher, "rs": args.rs_teacher, "uav": args.uav_teacher}, device=device)
        trainer = MultiTeacherDistillationTrainer(
            model, teachers, temperature=4.0, alpha=0.3, beta=0.7,
            aggregation="domain_routed" if args.kind == "kd_routed" else "weighted_avg", device=device)
        optimizer, _ = make_optimizer(trainer.student.parameters(), recipe, len(train_loader), args.epochs)

        def train_epoch(epoch: int) -> dict:
            m = trainer.train_epoch(train_loader, optimizer, epoch)
            return {"train_loss": m.loss, "train_hard_loss": m.hard_loss, "train_soft_loss": m.soft_loss, "train_acc": m.accuracy}
    else:
        optimizer, scheduler = make_optimizer(model.parameters(), recipe, len(train_loader), args.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=recipe.amp and device.type == "cuda")

        def train_epoch(epoch: int) -> dict:
            return train_supervised_epoch(model, train_loader, optimizer, scheduler, scaler, recipe, device,
                                          f"{args.kind} epoch {epoch}/{args.epochs}")

    metadata = {
        "kind": args.kind, "seed": args.seed, "epochs": args.epochs, "lr_factor": args.lr_factor,
        "plumes_in_training": not args.no_plumes, "balanced_rs_sampler": args.balanced_rs_sampler,
        "source_balanced_batches": args.source_balanced_batches,
        "limit_train": args.limit_train, "limit_val": args.limit_val, "train_sizes": sizes,
        "start_checkpoint": str(args.checkpoint), "start_checkpoint_sha256": sha256_file(args.checkpoint),
        "teachers": {k: str(v) for k, v in (("cv", args.cv_teacher), ("rs", args.rs_teacher), ("uav", args.uav_teacher)) if v},
        "plume_index_sha256": sha256_file(args.index_root / PLUME_INDEX),
        "git_commit": git_commit(ROOT), "git_dirty_files": dirty_files(), "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    if args.rs_teacher:
        metadata["rs_teacher_sha256"] = sha256_file(args.rs_teacher)
    run_finetune(model=model, base_checkpoint=base, val_loaders=val_loaders, train_epoch=train_epoch, recipe=recipe,
                 epochs=args.epochs, device=device, output_dir=args.output_dir, metadata=metadata)


if __name__ == "__main__":
    main()
