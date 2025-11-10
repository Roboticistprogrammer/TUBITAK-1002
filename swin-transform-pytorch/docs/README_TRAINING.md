# Swin Transformer for Fire and Smoke Detection

Train Swin Transformer on FASDD_RS dataset for fire and smoke detection using image classification.

## 📁 Project Structure

```
swin-transformer-pytorch/
├── config.py                    # Configuration file
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── requirements.txt             # Dependencies
├── swin_transformer_pytorch/
│   ├── __init__.py
│   ├── data_utils.py           # Dataset utilities
│   └── models/
│       ├── __init__.py
│       └── swin_transformer.py # Swin Transformer implementation
├── checkpoints/                 # Saved model checkpoints
├── logs/                        # TensorBoard logs
└── results/                     # Evaluation results
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Training

Edit `config.py` to customize your training:

```python
# Key configurations
DATASET_ROOT = "/mnt/storage/Dataset/FASDD_RS"
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
IMG_SIZE = 224
MODEL_VARIANT = 'tiny'  # 'tiny', 'small', 'base', 'large'
```

### 3. Train the Model

```bash
python train.py
```

The training script will:
- Load the FASDD_RS dataset
- Create a Swin Transformer model
- Train with mixed precision (if enabled)
- Save checkpoints and best model
- Log metrics to TensorBoard

### 4. Monitor Training

```bash
tensorboard --logdir logs/
```

### 5. Evaluate the Model

```bash
# Evaluate on test set
python evaluate.py --checkpoint checkpoints/best_model.pth --split test

# Evaluate with visualization
python evaluate.py --checkpoint checkpoints/best_model.pth --split test --visualize --save_plots
```

## ⚙️ Configuration Options

### Dataset Configuration

```python
ANNOTATION_FORMAT = 'coco'  # 'coco', 'voc', or 'yolo'
TASK = 'classification'     # 'classification' or 'detection'
NUM_CLASSES = 3             # fire, smoke, neitherFireNorSmoke
IMG_SIZE = 224              # Input image size (224, 384, 512)
```

### Model Variants

**Swin-T (Tiny)** - Default, fastest training
```python
HIDDEN_DIM = 96
LAYERS = (2, 2, 6, 2)
HEADS = (3, 6, 12, 24)
```

**Swin-S (Small)** - Better accuracy
```python
HIDDEN_DIM = 96
LAYERS = (2, 2, 18, 2)
HEADS = (3, 6, 12, 24)
```

**Swin-B (Base)** - High accuracy
```python
HIDDEN_DIM = 128
LAYERS = (2, 2, 18, 2)
HEADS = (4, 8, 16, 32)
```

**Swin-L (Large)** - Best accuracy
```python
HIDDEN_DIM = 192
LAYERS = (2, 2, 18, 2)
HEADS = (6, 12, 24, 48)
```

### Training Hyperparameters

```python
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
OPTIMIZER = 'adamw'          # 'adamw', 'adam', 'sgd'
SCHEDULER = 'cosine'         # 'cosine', 'step', 'none'
WARMUP_EPOCHS = 5
MIXED_PRECISION = True       # Use automatic mixed precision
GRADIENT_CLIP = 1.0          # Gradient clipping
LABEL_SMOOTHING = 0.1        # Label smoothing
```

### Regularization & Data Augmentation

The data augmentation is automatically applied:
- **Training**: RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomRotation
- **Validation/Test**: Only resize and normalize

### Class Imbalance

```python
USE_CLASS_WEIGHTS = True  # Automatically compute class weights
```

### Early Stopping

```python
EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for N epochs
```

## 📊 Outputs

### Training Outputs

- **Checkpoints**: `checkpoints/best_model.pth`, `checkpoints/checkpoint_epoch_N.pth`
- **TensorBoard Logs**: `logs/run_YYYYMMDD_HHMMSS/`
- **Training Metrics**: Loss, Accuracy, Learning Rate

### Evaluation Outputs

- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**: Visual representation of predictions
- **Per-Class Accuracy**: Bar chart showing accuracy for each class
- **Probability Distribution**: Confidence distribution for correct/incorrect predictions
- **Sample Predictions**: Visual examples with predictions

## 🔧 Advanced Usage

### Resume Training

```python
# In config.py
RESUME = True
RESUME_CHECKPOINT = "checkpoints/checkpoint_epoch_50.pth"
```

### Use Pretrained Weights

```python
# In config.py
PRETRAINED = True
PRETRAINED_PATH = "path/to/pretrained/weights.pth"
FREEZE_BACKBONE = True
UNFREEZE_AFTER_EPOCHS = 10
```

### Custom Training Script

```python
from config import Config
from swin_transformer_pytorch.models.swin_transformer import SwinTransformer
from swin_transformer_pytorch.data_utils import create_dataloaders

# Load config
config = Config()

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    root_dir=config.DATASET_ROOT,
    batch_size=config.BATCH_SIZE,
    img_size=config.IMG_SIZE,
    task=config.TASK
)

# Create model
model = SwinTransformer(
    hidden_dim=config.HIDDEN_DIM,
    layers=config.LAYERS,
    heads=config.HEADS,
    num_classes=config.NUM_CLASSES
)

# Your custom training loop...
```

## 📈 Expected Performance

On FASDD_RS dataset (classification task):

| Model   | Parameters | Image Size | Accuracy (Expected) |
|---------|-----------|------------|---------------------|
| Swin-T  | ~28M      | 224x224    | ~92-95%            |
| Swin-S  | ~50M      | 224x224    | ~93-96%            |
| Swin-B  | ~88M      | 224x224    | ~94-97%            |
| Swin-L  | ~197M     | 224x224    | ~95-98%            |

*Performance may vary based on dataset split and hyperparameters*

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in config.py
- Use smaller model variant (Swin-T instead of Swin-B)
- Reduce `IMG_SIZE`
- Set `MIXED_PRECISION = True`

### Training Too Slow
- Increase `NUM_WORKERS` for data loading
- Enable `MIXED_PRECISION`
- Set `CACHE_IMAGES = True` if you have enough RAM
- Use smaller `IMG_SIZE`

### Poor Accuracy
- Increase `EPOCHS`
- Adjust `LEARNING_RATE` (try 5e-5 or 2e-4)
- Enable `USE_CLASS_WEIGHTS` for imbalanced dataset
- Increase `IMG_SIZE` to 384 or 512
- Use larger model variant

## 📝 Citation

If you use this code, please cite the original Swin Transformer paper:

```bibtex
@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10012--10022},
  year={2021}
}
```

## 📄 License

MIT License
