"""Validation, checkpoint selection and the epoch loop of the plume fine-tuning (PLUME_PROTOCOL.md, Section 7).

Validation after every epoch (and once before fine-tuning, as epoch 0, which is never selected):
* CV and UAV val: four-class accuracy (models that have those domains);
* FASDD-RS val and plume val: binary smoke/fire-vs-neither metrics, logged separately, plus the pooled RS binary balanced
  accuracy over both sets together;
* selection score = mean of the CV accuracy, the UAV accuracy and the pooled RS binary balanced accuracy (for the RS teacher,
  which has no CV / UAV data, the pooled RS binary balanced accuracy alone);
* recall on the FASDD-RS val `both` images, for the pre-registered fallback (a drop of more than 2 points from epoch 0).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from firecls import binary_eval as be
from firecls.finetune.recipes import Recipe, lr_factor_at
from firecls.utils import save_json

FALLBACK_DROP_POINTS = 2.0


def logits_of(outputs) -> torch.Tensor:
    return getattr(outputs, "logits", outputs)


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for images, batch_labels, *_ in loader:
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            out = logits_of(model(images.to(device, non_blocking=True)))
        logits.append(out.float().cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def binary_balanced_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return be.binary_metrics(be.positive_prediction(logits), be.positive_score(logits), be.positive_truth(labels))["balanced_accuracy"]


def validation_record(model: nn.Module, val_loaders: Dict[str, DataLoader], device: torch.device, amp: bool) -> dict:
    record: dict = {}
    parts = {name: collect_logits(model, loader, device, amp) for name, loader in val_loaders.items()}
    components = []
    for key in ("cv", "uav"):
        if key in parts:
            logits, labels = parts[key]
            record[f"{key}_acc"] = float((logits.argmax(1) == labels).mean())
            components.append(record[f"{key}_acc"])
    for key in ("rs_fasdd", "rs_plume"):
        logits, labels = parts[key]
        metrics = be.binary_metrics(be.positive_prediction(logits), be.positive_score(logits), be.positive_truth(labels))
        record[key] = {k: metrics[k] for k in ("recall", "false_alarm_rate", "balanced_accuracy", "f1", "roc_auc", "errors")}
        record[key]["images"] = int(len(labels))
    fasdd_logits, fasdd_labels = parts["rs_fasdd"]
    record["rs_fasdd_both_recall"] = be.recall_on(be.positive_prediction(fasdd_logits), fasdd_labels, be.BOTH)["recall"]
    pooled_logits = np.concatenate([parts["rs_fasdd"][0], parts["rs_plume"][0]])
    pooled_labels = np.concatenate([parts["rs_fasdd"][1], parts["rs_plume"][1]])
    record["rs_pooled_binary_ba"] = binary_balanced_accuracy(pooled_logits, pooled_labels)
    components.append(record["rs_pooled_binary_ba"])
    # share of each predicted four-class label per validation set (a model that predicts one class for everything shows here)
    record["pred_share"] = {name: [float(x) for x in np.bincount(parts[name][0].argmax(1), minlength=4) / len(parts[name][0])]
                            for name in parts}
    record["selection_score"] = float(np.mean(components))
    return record


def train_supervised_epoch(model: nn.Module, loader: DataLoader, optimizer, scheduler, scaler, recipe: Recipe,
                           device: torch.device, description: str) -> dict:
    """One cross-entropy epoch with the model's own recipe (AMP, gradient clipping and per-step scheduler as originally)."""
    from tqdm import tqdm

    model.train()
    criterion = nn.CrossEntropyLoss()
    use_amp = recipe.amp and device.type == "cuda"
    total_loss, correct, seen = 0.0, 0, 0
    batch_losses = []
    progress = tqdm(loader, desc=description)
    for images, labels, *_ in progress:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=use_amp):
            logits = logits_of(model(images))
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        if recipe.grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), recipe.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * labels.size(0)
        batch_losses.append(loss.item())
        correct += int((logits.argmax(1) == labels).sum())
        seen += labels.size(0)
        progress.set_postfix(loss=total_loss / seen, acc=correct / seen)
    chunks = np.array_split(np.asarray(batch_losses), 10) if len(batch_losses) >= 10 else [np.asarray(batch_losses)]
    return {"train_loss": total_loss / max(1, seen), "train_acc": correct / max(1, seen),
            "loss_by_tenth_of_epoch": [float(c.mean()) for c in chunks if len(c)]}  # does the loss fall inside the epoch?


def make_optimizer(parameters, recipe: Recipe, steps_per_epoch: int, epochs: int):
    optimizer = torch.optim.AdamW(parameters, lr=recipe.lr, weight_decay=recipe.weight_decay)
    scheduler = None
    if recipe.schedule != "constant":  # restarted: step 0 of the fine-tuning is step 0 of the schedule
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: lr_factor_at(recipe, step, steps_per_epoch, epochs))
    return optimizer, scheduler


def checkpoint_payload(base: dict, model: nn.Module, epoch: int, score: float, finetune: dict) -> dict:
    """Same keys as the starting checkpoint (so every existing loader works), minus optimizer state and old history."""
    payload = {k: v for k, v in base.items() if k not in ("model", "optimizer", "scheduler", "history", "best_avg")}
    payload.update(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, epoch=epoch, avg_val=score,
                   finetune=finetune)
    return payload


def run_finetune(
    *,
    model: nn.Module,
    base_checkpoint: dict,
    val_loaders: Dict[str, DataLoader],
    train_epoch: Callable[[int], dict],
    recipe: Recipe,
    epochs: int,
    device: torch.device,
    output_dir: Path,
    metadata: dict,
) -> dict:
    """Epoch 0 (before) is evaluated, then `epochs` epochs are trained; best.pt = highest selection score over epochs 1..N."""
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    # Validation is FP32 whatever the training AMP setting, as in the original runs (their evaluation loops had no autocast).
    before = validation_record(model, val_loaders, device, False)
    before["epoch"] = 0
    history = [before]
    print(f"epoch 0 (before): score={before['selection_score']:.4f} rs_pooled_ba={before['rs_pooled_binary_ba']:.4f} "
          f"both_recall={before['rs_fasdd_both_recall']:.4f}", flush=True)
    best_score, best = -1.0, None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        metrics = train_epoch(epoch)
        record = validation_record(model, val_loaders, device, False)
        record.update(metrics, epoch=epoch, seconds=time.time() - t0)
        history.append(record)
        save_json(output_dir / "history.json", {"records": history, "recipe": recipe.asdict(), **metadata})
        finetune = {**metadata, "recipe": recipe.asdict(), "epoch": epoch, "validation": record}
        torch.save(checkpoint_payload(base_checkpoint, model, epoch, record["selection_score"], finetune), output_dir / "last.pt")
        if record["selection_score"] > best_score:
            best_score, best = record["selection_score"], record
            torch.save(checkpoint_payload(base_checkpoint, model, epoch, best_score, finetune), output_dir / "best.pt")
        print(f"epoch {epoch}: loss={metrics.get('train_loss', float('nan')):.4f} score={record['selection_score']:.4f} "
              f"rs_pooled_ba={record['rs_pooled_binary_ba']:.4f} both_recall={record['rs_fasdd_both_recall']:.4f} "
              f"best={best_score:.4f}", flush=True)
    drop = 100.0 * (before["rs_fasdd_both_recall"] - best["rs_fasdd_both_recall"])
    ba_drop = 100.0 * (before["rs_fasdd"]["balanced_accuracy"] - best["rs_fasdd"]["balanced_accuracy"])
    summary = {
        # RS-teacher fallback (PLUME_PROTOCOL.md, Section 7): FASDD-RS val binary balanced accuracy of the selected checkpoint
        # against the before model; a drop of more than 2 points means the teacher is rerun with source-balanced batches.
        "fasdd_rs_val_binary_ba_before": before["rs_fasdd"]["balanced_accuracy"],
        "fasdd_rs_val_binary_ba_best": best["rs_fasdd"]["balanced_accuracy"], "fasdd_rs_val_ba_drop_points": ba_drop,
        "ba_fallback_triggered": bool(ba_drop > FALLBACK_DROP_POINTS),
        "best_epoch": best["epoch"], "best_selection_score": best_score, "before_selection_score": before["selection_score"],
        "fasdd_rs_val_both_recall_before": before["rs_fasdd_both_recall"],
        "fasdd_rs_val_both_recall_best": best["rs_fasdd_both_recall"], "both_recall_drop_points": drop,
        "fallback_triggered": bool(drop > FALLBACK_DROP_POINTS), "seconds": time.time() - started,
        "peak_gpu_memory_gb": (torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else None,
    }
    save_json(output_dir / "summary.json", summary)
    print("summary:", summary, flush=True)
    return summary
