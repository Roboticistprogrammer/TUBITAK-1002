"""Train PIDNet-S on SAM pseudo-masks (Pesonen et al., WACV 2025), under the shared protocol.

Data: CV + RS + UAV ``train`` splits concatenated, as for the distilled student; masks from
``scripts/pidnet/generate_sam_masks.py``. Recipe (paper): AdamW, lr 1e-3, weight decay 1e-2,
batch 16, constant learning rate (no schedule is stated), PIDNet loss weights 0.4/20/1/1, t=0.8.
Budget (protocol): 20 epochs instead of the paper's 50, because the FASDD train split is ~19x larger than their
training set.

Model selection, every epoch on ``val``:
  * ``image_acc``: equal-weight mean over domains of image-level accuracy at thresholds 0.5/0.5
    (the protocol's rule; default);
  * ``miou``: equal-weight mean over domains of the sample-wise pseudo-label mIoU.
Both are logged to ``history.json``. ``--val-fraction`` < 1 scores a fixed, seeded subset of
each domain's val split per epoch for speed; thresholds are later calibrated on the full split.

    python scripts/pidnet/train_pidnet.py --seed 42 \
        --pretrained third_party/weights/PIDNet_S_ImageNet.pth.tar --amp
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import setup_path  # noqa: E402

ROOT = setup_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader, Subset  # noqa: E402

from firecls.baselines.data import build_domain_datasets  # noqa: E402
from firecls.baselines.pidnet.data import PseudoMaskDataset  # noqa: E402
from firecls.baselines.pidnet.losses import PidnetLoss  # noqa: E402
from firecls.baselines.pidnet.masks import mask_path  # noqa: E402
from firecls.baselines.pidnet.metrics import SegmentationMeter  # noqa: E402
from firecls.baselines.pidnet.model import (  # noqa: E402
    MODEL_NAME,
    SEG_CHANNELS,
    PidnetPresenceScores,
    build_pidnet_s,
    load_imagenet_pretrained,
    presence_from_logits,
    upsample_like,
)
from firecls.baselines.protocol import BATCH_SIZE, CLASSES, DOMAINS, EPOCHS  # noqa: E402
from firecls.baselines.scores import presence_to_scores  # noqa: E402
from firecls.deployment.artifacts import git_commit, write_json  # noqa: E402
from firecls.evaluation import classification_metrics  # noqa: E402
from firecls.utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=ROOT, help="directory that contains datasets/")
    parser.add_argument("--masks-root", type=Path, default=ROOT / "pseudo_masks")
    parser.add_argument("--output-dir", type=Path, default=None, help="default outputs/baselines/pidnet_s/seed<k>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=512, help="letterbox side; multiple of 8")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--topk-fraction", type=float, default=0.001, help="image presence = mean of top-K pixels")
    parser.add_argument("--pretrained", type=Path, default=None, help="PIDNet_S_ImageNet.pth.tar; random init if omitted")
    parser.add_argument("--amp", action="store_true", help="float16 autocast + GradScaler (CUDA only)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=1.0, help="fixed seeded subset of val per epoch")
    parser.add_argument("--select-by", choices=["image_acc", "miou"], default="image_acc")
    parser.add_argument("--resume", action="store_true", help="continue from <output-dir>/last.pt if present")
    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


def check_masks(datasets: dict[str, PseudoMaskDataset]) -> None:
    missing = [ds.mask_file(i) for ds in datasets.values() for i in range(len(ds)) if not ds.mask_file(i).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} pseudo-masks missing (first: {missing[0]}). Run scripts/pidnet/generate_sam_masks.py."
        )


def build_sets(args, split: str, augment: bool) -> dict[str, PseudoMaskDataset]:
    base = build_domain_datasets(split, None, args.dataset_root)
    return {
        domain: PseudoMaskDataset(base[domain].samples, domain, args.masks_root, args.img_size, augment=augment)
        for domain in DOMAINS
    }


def subsample(dataset, fraction: float, seed: int):
    if fraction >= 1.0:
        return dataset
    count = max(1, round(len(dataset) * fraction))
    indices = np.sort(np.random.default_rng(seed).choice(len(dataset), size=count, replace=False))
    return Subset(dataset, indices.tolist())


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def train_epoch(net, scorer, criterion, loader, optimizer, scaler, device, amp: bool, log_every: int, epoch: int) -> dict:
    net.train()
    sums = {"loss": 0.0, "loss_p": 0.0, "loss_boundary": 0.0, "loss_main": 0.0, "loss_bas": 0.0}
    steps = 0
    started = time.perf_counter()
    for step, (images, masks, edges, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        edges = edges.to(device, non_blocking=True)
        with autocast(device, amp):
            outputs = net(scorer.normalise(images))
        loss, parts = criterion(outputs, masks, edges)  # float32: outputs are cast inside the loss
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch} step {step}: {parts}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        sums["loss"] += loss_value
        for key, value in parts.items():
            sums[key] += value
        steps += 1
        if log_every and step % log_every == 0:
            print(f"  epoch {epoch} step {step}/{len(loader)} loss={sums['loss'] / steps:.4f}", flush=True)
    result = {key: value / max(1, steps) for key, value in sums.items()}
    result.update({"steps": steps, "seconds": time.perf_counter() - started})
    return result


@torch.no_grad()
def validate(net, scorer, loaders: dict[str, DataLoader], device, amp: bool) -> dict:
    """Image-level accuracy at 0.5/0.5 and pseudo-label IoU, per domain and domain-macro."""
    net.eval()
    domains = {}
    for domain, loader in loaders.items():
        meter = SegmentationMeter()
        labels, p_fire, p_smoke = [], [], []
        for images, masks, _, batch_labels in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with autocast(device, amp):
                logits = scorer.logits(images)
            logits = logits.float()
            fire, smoke = presence_from_logits(logits, scorer.k)
            meter.update(upsample_like(logits, masks.shape[-2:]), masks)
            p_fire.append(fire.cpu().numpy())
            p_smoke.append(smoke.cpu().numpy())
            labels.append(batch_labels.numpy())
        labels_np = np.concatenate(labels)
        predictions = presence_to_scores(np.concatenate(p_fire), np.concatenate(p_smoke), 0.5, 0.5).argmax(1)
        metrics = classification_metrics(labels_np, predictions, CLASSES)
        domains[domain] = {
            "samples": int(labels_np.size),
            "accuracy": metrics["accuracy"],
            "macro_f1_present_classes": metrics["macro_f1_present_classes"],
            **meter.compute(),
        }
    mious = [d["miou"] for d in domains.values() if d["miou"] is not None]
    return {
        "domains": domains,
        "image_acc": float(np.mean([d["accuracy"] for d in domains.values()])),
        "miou": float(np.mean(mious)) if mious else None,
    }


def atomic_save(payload: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def plain_args(args: argparse.Namespace) -> dict:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or ROOT / "outputs" / "baselines" / MODEL_NAME / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_sets = build_sets(args, "train", augment=True)
    val_sets = build_sets(args, "val", augment=False)
    check_masks(train_sets)
    check_masks(val_sets)
    train_data = ConcatDataset(list(train_sets.values()))
    val_loaders = {
        domain: DataLoader(
            subsample(ds, args.val_fraction, args.seed), batch_size=args.val_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        for domain, ds in val_sets.items()
    }
    print(f"train images: {len(train_data)}; val images/epoch: "
          f"{ {d: len(l.dataset) for d, l in val_loaders.items()} }")

    net = build_pidnet_s(len(SEG_CHANNELS), augment=True)
    if args.pretrained:
        init = {"type": "imagenet", **load_imagenet_pretrained(net, args.pretrained)}
        print(f"Loaded {init['tensors_loaded']}/{init['tensors_total']} tensors from {args.pretrained}")
    else:
        init = {"type": "random"}
        print("WARNING: random initialisation; Pesonen et al. start from ImageNet weights (--pretrained).")
    scorer = PidnetPresenceScores(net, args.img_size, args.topk_fraction).to(device)
    criterion = PidnetLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    history: list[dict] = []
    best_value, best_epoch, start_epoch = -math.inf, None, 1
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        for key in ("img_size", "topk_fraction", "select_by", "seed", "val_fraction"):
            if state.get(key) != getattr(args, key):
                raise SystemExit(f"--resume: {key}={getattr(args, key)} differs from {last_path} ({state.get(key)})")
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history, best_value, best_epoch = state["history"], state["best_value"], state["best_epoch"]
        init = state["init"]
        start_epoch = state["epoch"] + 1
        print(f"Resumed from {last_path} at epoch {start_epoch}")

    base = {
        "classes": list(CLASSES),
        "seg_channels": list(SEG_CHANNELS),
        "model_name": MODEL_NAME,
        "img_size": args.img_size,
        "pad_value": 114,
        "topk_fraction": args.topk_fraction,
        "topk_k": scorer.k,
        "seed": args.seed,
        "select_by": args.select_by,
        "val_fraction": args.val_fraction,
        "init": init,
        "args": plain_args(args),
        "git_commit": git_commit(ROOT),
    }
    for epoch in range(start_epoch, args.epochs + 1):
        # Reseed per epoch so a resumed run sees the same shuffling/augmentation as an unbroken one.
        torch.manual_seed(args.seed * 1000 + epoch)
        generator = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
            pin_memory=True, drop_last=True, generator=generator,
        )
        train_metrics = train_epoch(
            net, scorer, criterion, train_loader, optimizer, scaler, device, args.amp, args.log_every, epoch
        )
        val = validate(net, scorer, val_loaders, device, args.amp)
        value = val[args.select_by]
        value = -math.inf if value is None else value
        improved = value > best_value
        if improved:
            best_value, best_epoch = value, epoch
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val,
            "selected_metric": args.select_by,
            "selected_value": val[args.select_by],
            "is_best": improved,
        }
        history.append(record)
        write_json(output_dir / "history.json", {"config": base, "records": history, "best_epoch": best_epoch})
        checkpoint = {**base, "model": net.state_dict(), "epoch": epoch, "selection_value": val[args.select_by],
                      "history": history}
        if improved:
            atomic_save(checkpoint, best_path)
        atomic_save(
            {**checkpoint, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
             "best_value": best_value, "best_epoch": best_epoch},
            last_path,
        )
        miou = "n/a" if val["miou"] is None else f"{val['miou']:.4f}"
        print(
            f"Epoch {epoch}: loss={train_metrics['loss']:.4f} val_image_acc={val['image_acc']:.4f} "
            f"val_miou={miou} {'*' if improved else ''}",
            flush=True,
        )

    write_json(
        output_dir / "done.json",
        {"best_epoch": best_epoch, "best_value": best_value if math.isfinite(best_value) else None, "select_by": args.select_by, "epochs": args.epochs,
         "finished_at": datetime.now(timezone.utc).isoformat()},
    )
    print(f"Best epoch {best_epoch} ({args.select_by}={best_value:.4f}): {best_path}")


if __name__ == "__main__":
    main()
