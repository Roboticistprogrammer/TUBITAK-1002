"""
Training script for FASDD_RS Fire and Smoke Detection using Swin Transformer
"""

import os
import sys
import time
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from swin_transformer_pytorch.models.swin_transformer import SwinTransformer
from swin_transformer_pytorch.data_utils import create_dataloaders, FASSDDataset


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsTracker:
    """Track training metrics"""
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def get_latest(self, key):
        return self.metrics[key][-1] if self.metrics[key] else 0
    
    def get_average(self, key, last_n=None):
        values = self.metrics[key]
        if not values:
            return 0
        if last_n:
            values = values[-last_n:]
        return sum(values) / len(values)


def create_model(config):
    """Create Swin Transformer model"""
    print(f"\nCreating Swin Transformer model...")
    
    model = SwinTransformer(
        hidden_dim=config.HIDDEN_DIM,
        layers=config.LAYERS,
        heads=config.HEADS,
        channels=config.CHANNELS,
        num_classes=config.NUM_CLASSES,
        head_dim=config.HEAD_DIM,
        window_size=config.WINDOW_SIZE,
        downscaling_factors=config.DOWNSCALING_FACTORS,
        relative_pos_embedding=config.RELATIVE_POS_EMBEDDING
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def create_optimizer(model, config):
    """Create optimizer"""
    if config.OPTIMIZER.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.BETAS
        )
    elif config.OPTIMIZER.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.BETAS
        )
    elif config.OPTIMIZER.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            momentum=config.MOMENTUM
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
    
    print(f"Optimizer: {config.OPTIMIZER.upper()}")
    return optimizer


def create_scheduler(optimizer, config, steps_per_epoch):
    """Create learning rate scheduler"""
    if config.SCHEDULER == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.EPOCHS - config.WARMUP_EPOCHS,
            eta_min=config.MIN_LR
        )
        print(f"Scheduler: Cosine Annealing (warmup={config.WARMUP_EPOCHS} epochs)")
    elif config.SCHEDULER == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.STEP_SIZE,
            gamma=config.GAMMA
        )
        print(f"Scheduler: Step LR (step_size={config.STEP_SIZE}, gamma={config.GAMMA})")
    else:
        scheduler = None
        print("Scheduler: None")
    
    return scheduler


def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    """Warmup learning rate"""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    return None


def compute_class_weights(dataset, config):
    """Compute class weights for imbalanced dataset"""
    print("\nComputing class weights...")
    class_counts = defaultdict(int)
    
    for sample in dataset.samples:
        label = sample.get('image_label', 'neitherFireNorSmoke')
        class_counts[label] += 1
    
    total = sum(class_counts.values())
    weights = []
    
    for class_name in config.CLASS_NAMES:
        count = class_counts.get(class_name, 1)
        weight = total / (len(config.CLASS_NAMES) * count)
        weights.append(weight)
    
    print("Class distribution:")
    for name, weight in zip(config.CLASS_NAMES, weights):
        count = class_counts.get(name, 0)
        print(f"  {name}: {count} samples (weight: {weight:.4f})")
    
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, dataloader, criterion, optimizer, device, config, epoch, scaler=None, writer=None):
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.EPOCHS} [Train]")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        batch_size = images.size(0)
        
        # Forward pass with mixed precision
        if config.MIXED_PRECISION and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Compute accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        accuracy = 100. * correct / batch_size
        
        # Backward pass
        optimizer.zero_grad()
        
        if config.MIXED_PRECISION and scaler is not None:
            scaler.scale(loss).backward()
            if config.GRADIENT_CLIP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.GRADIENT_CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
        
        # Update metrics
        losses.update(loss.item(), batch_size)
        accuracies.update(accuracy, batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Log to tensorboard
        if writer and batch_idx % config.LOG_INTERVAL == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
            writer.add_scalar('Train/Accuracy_Step', accuracy, global_step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)
    
    return losses.avg, accuracies.avg


@torch.no_grad()
def validate(model, dataloader, criterion, device, config, epoch=0):
    """Validate model"""
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    # Per-class metrics
    class_correct = [0] * config.NUM_CLASSES
    class_total = [0] * config.NUM_CLASSES
    
    all_predictions = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.EPOCHS} [Val]")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        batch_size = images.size(0)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Compute accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        accuracy = 100. * correct / batch_size
        
        # Update metrics
        losses.update(loss.item(), batch_size)
        accuracies.update(accuracy, batch_size)
        
        # Per-class accuracy
        for label, pred in zip(labels, predicted):
            class_total[label] += 1
            if label == pred:
                class_correct[label] += 1
        
        # Store for confusion matrix
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.2f}%'
        })
    
    # Compute per-class accuracy
    class_accuracies = {}
    for i, class_name in enumerate(config.CLASS_NAMES):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            class_accuracies[class_name] = acc
    
    return losses.avg, accuracies.avg, class_accuracies, (all_predictions, all_labels)


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, config, filename='checkpoint.pth'):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric,
        'config': {
            'HIDDEN_DIM': config.HIDDEN_DIM,
            'LAYERS': config.LAYERS,
            'HEADS': config.HEADS,
            'NUM_CLASSES': config.NUM_CLASSES,
            'HEAD_DIM': config.HEAD_DIM,
            'WINDOW_SIZE': config.WINDOW_SIZE,
            'DOWNSCALING_FACTORS': config.DOWNSCALING_FACTORS,
            'RELATIVE_POS_EMBEDDING': config.RELATIVE_POS_EMBEDDING,
            'CHANNELS': config.CHANNELS,
            'IMG_SIZE': config.IMG_SIZE,
        }
    }
    
    save_path = config.CHECKPOINT_DIR / filename
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load checkpoint"""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    best_metric = checkpoint['best_metric']
    
    print(f"Resumed from epoch {epoch}, best metric: {best_metric:.4f}")
    return epoch, best_metric


def main():
    """Main training function"""
    # Load configuration
    config = Config()
    config.create_dirs()
    config.display()
    
    # Set seed
    set_seed(config.SEED)
    
    # Device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create dataloaders
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=config.DATASET_ROOT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        img_size=config.IMG_SIZE,
        annotation_format=config.ANNOTATION_FORMAT,
        task=config.TASK,
        pin_memory=config.PIN_MEMORY,
        cache_images=config.CACHE_IMAGES
    )
    
    print(f"\nDataset loaded successfully!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
    model = create_model(config)
    model = model.to(device)
    
    # Compile model (PyTorch 2.0+)
    if config.COMPILE_MODEL and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    
    # Loss function
    if config.USE_CLASS_WEIGHTS:
        train_dataset = FASSDDataset(
            root_dir=config.DATASET_ROOT,
            split='train',
            annotation_format=config.ANNOTATION_FORMAT,
            task=config.TASK
        )
        class_weights = compute_class_weights(train_dataset, config)
        class_weights = class_weights.to(device)
    else:
        class_weights = None
    
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.LABEL_SMOOTHING
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.MIXED_PRECISION else None
    
    # TensorBoard writer
    writer = None
    if config.USE_TENSORBOARD:
        log_dir = config.LOGS_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir)
        print(f"\nTensorBoard logs: {log_dir}")
    
    # Resume from checkpoint
    start_epoch = 1
    best_metric = 0.0 if config.VAL_MODE == 'max' else float('inf')
    epochs_no_improve = 0
    
    if config.RESUME and config.RESUME_CHECKPOINT:
        start_epoch, best_metric = load_checkpoint(
            model, optimizer, scheduler, config.RESUME_CHECKPOINT
        )
        start_epoch += 1
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    metrics_tracker = MetricsTracker()
    
    for epoch in range(start_epoch, config.EPOCHS + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config.EPOCHS}")
        print(f"{'='*70}")
        
        # Warmup
        if epoch <= config.WARMUP_EPOCHS:
            warmup_lr(optimizer, epoch - 1, config.WARMUP_EPOCHS, config.LEARNING_RATE)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, config,
            epoch, scaler, writer
        )
        
        # Validate
        if epoch % config.EVAL_INTERVAL == 0:
            val_loss, val_acc, class_accs, _ = validate(
                model, val_loader, criterion, device, config, epoch
            )
            
            print(f"\n[Epoch {epoch}] Results:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"  Per-class accuracy:")
            for class_name, acc in class_accs.items():
                print(f"    {class_name}: {acc:.2f}%")
            
            # Update metrics
            metrics_tracker.update(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc
            )
            
            # TensorBoard logging
            if writer:
                writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
                writer.add_scalar('Train/Accuracy_Epoch', train_acc, epoch)
                writer.add_scalar('Val/Loss', val_loss, epoch)
                writer.add_scalar('Val/Accuracy', val_acc, epoch)
                for class_name, acc in class_accs.items():
                    writer.add_scalar(f'Val/Accuracy_{class_name}', acc, epoch)
            
            # Check if best model
            current_metric = val_acc if config.VAL_METRIC == 'accuracy' else val_loss
            is_best = False
            
            if config.VAL_MODE == 'max':
                if current_metric > best_metric:
                    best_metric = current_metric
                    is_best = True
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                if current_metric < best_metric:
                    best_metric = current_metric
                    is_best = True
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            
            # Save checkpoint
            if is_best:
                print(f"\n✓ New best model! {config.VAL_METRIC}: {best_metric:.4f}")
                save_checkpoint(model, optimizer, scheduler, epoch, best_metric, config, 'best_model.pth')
            
            if not config.SAVE_BEST_ONLY and epoch % config.SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, best_metric, config, f'checkpoint_epoch_{epoch}.pth')
            
            # Early stopping
            if config.EARLY_STOPPING_PATIENCE and epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered! No improvement for {config.EARLY_STOPPING_PATIENCE} epochs.")
                break
        
        # Step scheduler
        if scheduler and epoch > config.WARMUP_EPOCHS:
            scheduler.step()
    
    # Final evaluation on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    # Load best model
    best_checkpoint_path = config.CHECKPOINT_DIR / 'best_model.pth'
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    test_loss, test_acc, test_class_accs, (predictions, labels) = validate(
        model, test_loader, criterion, device, config, epoch
    )
    
    print(f"\n[Test Results]")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  Per-class accuracy:")
    for class_name, acc in test_class_accs.items():
        print(f"    {class_name}: {acc:.2f}%")
    
    # Save results
    results_file = config.RESULTS_DIR / 'test_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
        f.write("Per-class Accuracy:\n")
        for class_name, acc in test_class_accs.items():
            f.write(f"  {class_name}: {acc:.2f}%\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Close writer
    if writer:
        writer.close()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"Best {config.VAL_METRIC}: {best_metric:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == '__main__':
    main()
