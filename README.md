# Swin Transformer (Small) - Fire & Smoke Detection

Training pipeline for fire and smoke detection using Swin Transformer architecture on the FASDD_RS dataset.

## 🎯 Current Model Configuration

### Architecture Parameters
```python
net = SwinTransformer(
    hidden_dim=96,           # Base channel dimension
    layers=(2, 2, 6, 2),     # Blocks per stage
    heads=(3, 6, 12, 24),    # Attention heads per stage
    channels=3,              # Input channels (RGB)
    num_classes=3,           # [fire, smoke, neitherFireNorSmoke]
    head_dim=32,             # Dimension per attention head
    window_size=7,           # Local attention window size
    downscaling_factors=(4, 2, 2, 2),  # Spatial reduction per stage
    relative_pos_embedding=True  # Relative positional embeddings
)
```

**Model Type**: Swin-S (Small) - ~50M parameters
- Input: (B, 3, 224, 224)
- Output: (B, 3) logits

### Training Hyperparameters
```python
# Data
BATCH_SIZE = 16
IMG_SIZE = 224
NUM_WORKERS = 8
CACHE_IMAGES = False

# Optimization
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
OPTIMIZER = 'adamw'

# Scheduling
SCHEDULER = 'cosine'
WARMUP_EPOCHS = 5
MIN_LR = 1e-6

# Regularization
LABEL_SMOOTHING = 0.1
GRADIENT_CLIP = 1.0
DROPOUT = 0.0

# Strategy
MIXED_PRECISION = True
EARLY_STOPPING_PATIENCE = 15
```

### Dataset Configuration
```python
# Multi-source training
USE_MULTI_DATASETS = True
TRAIN_DATASETS = ['FASDD_RS', 'FASDD_CV', 'FASDD_UAV']
VAL_DATASETS = ['FASDD_RS', 'FASDD_CV', 'FASDD_UAV']
TEST_DATASETS = ['FASDD_RS', 'FASDD_CV', 'FASDD_UAV']

# Classes
NUM_CLASSES = 3
CLASS_NAMES = ['fire', 'smoke', 'neitherFireNorSmoke']
ANNOTATION_FORMAT = 'coco'
TASK = 'classification'
```

## 🚀 Quick Start

### 1. Training
```bash
# Navigate to the project directory first
cd swin-transform-pytorch

# Train with current config
python train.py

# Custom configuration: Edit config.py first, then:
python train.py
```

### 2. Evaluation
```bash
cd swin-transform-pytorch

# Evaluate on test set
python evaluate.py --checkpoint checkpoints/best_model.pth --split test --visualize
```

### 3. Inference on Single Image
```bash
cd swin-transform-pytorch

# Basic inference (prints results to console)
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image results/example/fire.jpg

# With visualization
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image results/example/fire.jpg \
    --visualize

# Save visualization to file
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image /results/example/fire.jpg \
    --visualize \
    --save results/example/prediction.png

# Using CPU instead of GPU
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image results/example/fire.jpg \
    --device cpu \
    --visualize
```

## 📊 Model Variants Quick Reference

| Variant | Hidden Dim | Layers       | Heads        | Parameters | Speed  |
|---------|-----------|--------------|--------------|-----------|--------|
| Swin-T  | 96        | (2,2,6,2)    | (3,6,12,24)  | ~28M      | Fast   |
| Swin-S  | 96        | (2,2,18,2)   | (3,6,12,24)  | ~50M      | Medium |
| Swin-B  | 128       | (2,2,18,2)   | (4,8,16,32)  | ~88M      | Slow   |
| Swin-L  | 192       | (2,2,18,2)   | (6,12,24,48) | ~197M     | Slowest|

To use a different variant, edit `swin-transform-pytorch/config.py` and set `MODEL_VARIANT` or uncomment the desired configuration.

## 📝 Parameter Tracking for New Models

When training new models with different parameters, document them here:

### Model Run Template
```markdown
#### Run: [Description] - [Date]
- **Config File**: config.py
- **Model Type**: Swin-T / Swin-S / Swin-B / Swin-L
- **Key Parameters**:
  - Image Size: 224 / 384 / 512
  - Batch Size: X
  - Learning Rate: X
  - Epochs: X
  - Scheduler: cosine / step / none
  - Mixed Precision: Yes / No
  - Label Smoothing: X
- **Datasets**: FASDD_RS / Combined (RS+CV+UAV)
- **Results**:
  - Best Accuracy: X%
  - Best Loss: X
  - Checkpoint: checkpoints/best_model_[run_name].pth
  - Notes: [Any observations]
```

### Example Run Documentation

#### Run: Baseline - Swin-T on FASDD_RS (2024-11)
- **Model Type**: Swin-T (28M params)
- **Key Parameters**:
  - Image Size: 224
  - Batch Size: 16
  - Learning Rate: 1e-4
  - Epochs: 100
  - Scheduler: Cosine with warmup (5 epochs)
  - Mixed Precision: Yes
  - Label Smoothing: 0.1
- **Datasets**: Combined (FASDD_RS, FASDD_CV, FASDD_UAV)
- **Status**: Ready to train
- **Expected Time**: ~3.3 hours on RTX 3090

---

#### Run: [Next experiment] - [Your description]
- **Model Type**: 
- **Key Parameters**:
  - Image Size: 
  - Batch Size: 
  - Learning Rate: 
  - Epochs: 
  - Scheduler: 
  - Mixed Precision: 
  - Label Smoothing: 
- **Datasets**: 
- **Status**: [Planned / In Progress / Complete]
- **Results**: [To be filled after training]

---

## 📁 Directory Structure

```
e:\Amir_Thesis\TUBITAK-1002\
├── swin-transform-pytorch/
│   ├── config.py                    # ← Modify parameters here
│   ├── train.py                     # Main training script
│   ├── evaluate.py                  # Evaluation & metrics
│   ├── inference.py                 # Single image prediction
│   ├── data_utils.py                # Dataset loading
│   ├── swin_transformer.py          # Model implementation
│   ├── checkpoints/                 # Saved models
│   ├── logs/                        # TensorBoard logs
│   └── results/                     # Evaluation results
├── datasets/
│   ├── FASDD_RS/
│   ├── FASDD_CV/
│   └── FASDD_UAV/
└── README.md                        # This file
```

## 🔧 Modifying Parameters

1. **Architecture**: Edit `HIDDEN_DIM`, `LAYERS`, `HEADS` in `swin-transform-pytorch/config.py`
2. **Data**: Edit `IMG_SIZE`, `BATCH_SIZE`, `TRAIN_DATASETS` in `swin-transform-pytorch/config.py`
3. **Training**: Edit `EPOCHS`, `LEARNING_RATE`, `SCHEDULER` in `swin-transform-pytorch/config.py`
4. **Regularization**: Edit `LABEL_SMOOTHING`, `WEIGHT_DECAY`, `DROPOUT` in `swin-transform-pytorch/config.py`

See [TRAINING_GUIDE.md](swin-transform-pytorch/docs/TRAINING_GUIDE.md) for detailed explanation of each parameter.

## 🛠️ Terminal Tips

### For Bash (Git Bash, WSL, Linux, macOS)
Use **forward slashes** (`/`):
```bash
cd swin-transform-pytorch
python inference.py --checkpoint checkpoints/best_model.pth --image ../results/example/fire.jpg
```

### For Windows CMD/PowerShell
Use **backslashes** (`\`) or **forward slashes** (`/`):
```cmd
cd swin-transform-pytorch
python inference.py --checkpoint checkpoints\best_model.pth --image ..\results\example\fire.jpg
```

**Tip**: Forward slashes work in ALL terminals (bash, cmd, PowerShell), so always use `/` for cross-platform compatibility!

## 📚 Resources

- [Training Guide](swin-transform-pytorch/docs/TRAINING_GUIDE.md) - Detailed setup, tips, and troubleshooting
- Original Paper: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2105.01601)

