# Swin Transformer Training Guide - Base Architecture

Complete guide for training the Swin Transformer Base model on fire and smoke detection datasets.

## Current Configuration: Swin-B (Base)

This branch is configured for **Swin Transformer Base** architecture training.

### Model Specs
- **Architecture**: Swin-B
- **Parameters**: ~88M
- **Hidden Dim**: 128
- **Layers**: (2, 2, 18, 2)
- **Heads**: (4, 8, 16, 32)
- **Input Size**: 224x224

## Prerequisites

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Ensure datasets are organized as follows:

```
datasets/
├── FASDD_RS/
│   ├── train/
│   │   ├── images/
│   │   └── annotations.json
│   ├── val/
│   └── test/
├── FASDD_CV/
│   └── FASDD_CV/
│       ├── train/
│       ├── val/
│       └── test/
└── FASDD_UAV/
    ├── train/
    ├── val/
    └── test/
```

### 3. Verify Dataset Configuration

```bash
python test_datasets.py
```

This validates:
- Dataset paths exist
- COCO annotations are valid
- All 3 classes present
- Images loadable

## Training Workflow

### Step 1: Review Configuration

Check [config.py](../config.py) settings:

```python
# Verify Swin-B is active
MODEL_VARIANT = 'base'
HIDDEN_DIM = 128
LAYERS = (2, 2, 18, 2)
HEADS = (4, 8, 16, 32)

# Verify multi-dataset training
USE_MULTI_DATASETS = True
TRAIN_DATASETS = ['FASDD_RS', 'FASDD_CV', 'FASDD_UAV']
```

### Step 2: Display Configuration

```bash
python -c "from config import Config; Config.display()"
```

This prints all active settings for verification.

### Step 3: Start Training

```bash
# Basic training
python train.py

# With specific GPU
CUDA_VISIBLE_DEVICES=0 python train.py

# With logging level
python train.py --log-level INFO
```

### Step 4: Monitor Training

#### TensorBoard (Default)
```bash
tensorboard --logdir=logs/
```

Navigate to `http://localhost:6006`

#### Weights & Biases (Optional)
```python
# In config.py
USE_WANDB = True
WANDB_PROJECT = "fasdd-fire-smoke-detection"
WANDB_ENTITY = "your-username"
```

## Training Stages

### Warmup Phase (Epochs 1-5)
- Learning rate gradually increases from 0 → 1e-4
- Model stabilizes
- Loss decreases rapidly

### Main Training (Epochs 6-85)
- Cosine annealing of learning rate
- Steady improvement in accuracy
- Monitor validation metrics

### Fine-tuning (Epochs 86-100)
- Very low learning rate (~1e-6)
- Minor refinements
- Early stopping may trigger

## Expected Behavior

### Loss Curves
- **Training Loss**: Should decrease smoothly
- **Validation Loss**: Should track training loss
- **Gap**: Small gap indicates good generalization

### Accuracy Progression
- **Epoch 10**: ~70-75% validation accuracy
- **Epoch 30**: ~85-88% validation accuracy
- **Epoch 60**: ~90-92% validation accuracy
- **Epoch 100**: ~93-95% validation accuracy (target)

### Checkpointing
Best model saved to:
```
checkpoints/best_model_swin_base.pth
```

Regular checkpoints (every 5 epochs):
```
checkpoints/checkpoint_epoch_5.pth
checkpoints/checkpoint_epoch_10.pth
...
```

## Advanced Options

### Resume Training

If training interrupted:

```python
# In config.py
RESUME = True
RESUME_CHECKPOINT = 'checkpoints/checkpoint_epoch_45.pth'
```

```bash
python train.py
```

### Transfer Learning

Use ImageNet pre-trained weights:

```python
# In config.py
PRETRAINED = True
PRETRAINED_PATH = 'pretrained/swin_base_patch4_window7_224.pth'
FREEZE_BACKBONE = True
UNFREEZE_AFTER_EPOCHS = 10
```

### Adjust Batch Size

For memory constraints:

```python
# In config.py
BATCH_SIZE = 8  # Reduce from 16
```

**Note**: Halving batch size → double the training time

### Mixed Precision Training

Already enabled by default:

```python
MIXED_PRECISION = True  # Uses AMP for 30-40% speedup
```

To disable (for debugging):

```python
MIXED_PRECISION = False
```

## Troubleshooting

### Out of Memory (OOM)

**Solution 1**: Reduce batch size
```python
BATCH_SIZE = 8  # or 4
```

**Solution 2**: Reduce image size
```python
IMG_SIZE = 192  # from 224
```

**Solution 3**: Use gradient accumulation
```python
GRADIENT_ACCUMULATION_STEPS = 2
```

### Poor Convergence

**Check**:
1. Learning rate too high/low
2. Batch size too small
3. Data augmentation too aggressive
4. Class imbalance

**Solutions**:
```python
# Reduce LR
LEARNING_RATE = 5e-5

# Use class weights
USE_CLASS_WEIGHTS = True

# Increase warmup
WARMUP_EPOCHS = 10
```

### Overfitting

**Symptoms**: Large train-val accuracy gap

**Solutions**:
```python
# Increase regularization
DROPOUT = 0.1
LABEL_SMOOTHING = 0.2
WEIGHT_DECAY = 0.1

# Add augmentation in data_utils.py
```

### Slow Training

**Check**:
1. `NUM_WORKERS` setting
2. Data loading bottleneck
3. Mixed precision enabled

**Solutions**:
```python
NUM_WORKERS = 8  # Match CPU cores
PIN_MEMORY = True
USE_CUDNN_BENCHMARK = True
MIXED_PRECISION = True
```

## Post-Training

### Evaluation

```bash
# Evaluate best model
python evaluate.py --checkpoint checkpoints/best_model_swin_base.pth

# Evaluate on specific dataset
python evaluate.py --dataset FASDD_RS

# Evaluate on all datasets
python evaluate.py --all-datasets
```

### Inference

```bash
# Single image
python inference.py --image path/to/image.jpg

# Batch inference
python inference.py --image-dir path/to/images/

# With visualization
python inference.py --image path/to/image.jpg --visualize
```

## Performance Benchmarks

### Swin-B Expected Performance

| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| FASDD_RS | 93-95% | 92-94% | 91-93% | 92-94% |
| FASDD_CV | 94-96% | 93-95% | 92-94% | 93-95% |
| FASDD_UAV | 91-93% | 90-92% | 89-91% | 90-92% |
| **Combined** | **93-95%** | **92-94%** | **91-93%** | **92-94%** |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1080 Ti (11GB) | RTX 3090 (24GB) |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB | 100 GB |
| CPU | 8 cores | 16 cores |

### Training Time Estimates

| GPU | Batch Size | Time/Epoch | Total (100 epochs) |
|-----|------------|------------|-------------------|
| RTX 3090 | 16 | 15-18 min | ~25-30 hours |
| A100 | 16 | 10-12 min | ~17-20 hours |
| V100 | 16 | 18-22 min | ~30-37 hours |
| RTX 3080 | 8 | 20-25 min | ~33-42 hours |

## Configuration Files

All settings in [`Config`](../config.py) class in [config.py](../config.py):

- Model architecture: Lines 56-86
- Training hyperparameters: Lines 88-102
- Dataset paths: Lines 21-43
- Logging & checkpointing: Lines 116-125

## Next Steps

1. ✅ Verify configuration: Swin-B active
2. ✅ Test datasets: `python test_datasets.py`
3. ✅ Start training: `python train.py`
4. ✅ Monitor progress: TensorBoard
5. ✅ Evaluate results: `python evaluate.py`
6. ✅ Run inference: `python inference.py`

## Branch-Specific Notes

This **Swin-B branch** is optimized for:
- **Production deployment**: Balanced speed/accuracy
- **Multi-dataset training**: All 3 FASDD variants
- **Transfer learning**: Compatible with ImageNet pre-trained weights
- **Research baseline**: Strong baseline for comparisons

For other architectures, switch branches:
- `swin-tiny`: Lightweight, faster inference
- `swin-large`: Maximum accuracy, slower training