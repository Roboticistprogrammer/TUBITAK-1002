# Training Pipeline Summary

## 📋 Complete Training Setup

Your Swin Transformer training pipeline is now ready! Here's what has been created:

### 🗂️ Core Files

1. **`config.py`** - Central configuration file
   - All hyperparameters in one place
   - Easy to modify without changing code
   - Supports multiple model variants

2. **`train.py`** - Main training script
   - Uses ONLY your Swin Transformer implementation
   - Supports mixed precision training
   - Automatic checkpointing and early stopping
   - TensorBoard logging
   - Handles class imbalance
   - Warmup + Cosine/Step LR scheduling

3. **`evaluate.py`** - Comprehensive evaluation
   - Detailed metrics (Accuracy, Precision, Recall, F1)
   - Confusion matrix
   - Per-class accuracy
   - Probability distribution analysis
   - Sample visualizations

4. **`inference.py`** - Single image prediction
   - Easy-to-use interface
   - Probability visualization
   - Batch prediction support

5. **`swin_transformer_pytorch/data_utils.py`** - Optimized dataset utilities
   - Supports COCO, VOC, YOLO formats
   - Classification & detection tasks
   - Efficient data loading
   - Built-in augmentation
   - Image caching option

## 🚀 Usage Examples

### Basic Training
```bash
# Default configuration (Swin-T, 224x224, 100 epochs)
python train.py
```

### Custom Configuration
Edit `config.py` first:
```python
BATCH_SIZE = 32        # Increase if you have more GPU memory
EPOCHS = 150           # Train longer
IMG_SIZE = 384         # Higher resolution
MODEL_VARIANT = 'small'  # Use Swin-S
LEARNING_RATE = 5e-5   # Adjust learning rate
```

Then run:
```bash
python train.py
```

### Resume Training
```python
# In config.py
RESUME = True
RESUME_CHECKPOINT = "checkpoints/checkpoint_epoch_50.pth"
```

### Evaluation
```bash
# Basic evaluation
python evaluate.py --checkpoint checkpoints/best_model.pth --split test

# With visualizations
python evaluate.py --checkpoint checkpoints/best_model.pth --split test --visualize --save_plots
```

### Single Image Prediction
```bash
python inference.py --checkpoint checkpoints/best_model.pth \
                    --image path/to/image.tif \
                    --visualize \
                    --save results/prediction.png
```

## 📊 Training Features

### ✅ Implemented Features

- **Pure Swin Transformer**: Uses ONLY your implementation, no external dependencies
- **Mixed Precision**: Faster training with AMP (Automatic Mixed Precision)
- **Smart Scheduling**: Warmup + Cosine/Step decay
- **Data Augmentation**: RandomFlip, ColorJitter, Rotation
- **Class Balancing**: Automatic class weight computation
- **Early Stopping**: Prevents overfitting
- **Gradient Clipping**: Training stability
- **Label Smoothing**: Regularization technique
- **Checkpoint Management**: Auto-save best model
- **TensorBoard Logging**: Real-time monitoring
- **Multi-format Support**: COCO, VOC, YOLO annotations

### 🎯 Training Loop Structure

```
For each epoch:
  1. Warmup LR (first N epochs)
  2. Training phase:
     - Forward pass
     - Compute loss (with label smoothing)
     - Backward pass (with mixed precision)
     - Gradient clipping
     - Optimizer step
     - Log metrics every N batches
  3. Validation phase:
     - Evaluate on validation set
     - Compute per-class metrics
     - Save best model
     - Check early stopping
  4. LR scheduling (after warmup)
```

## 🔧 Configuration Guide

### Model Selection

| Variant | Parameters | Speed    | Accuracy | Recommended For          |
|---------|-----------|----------|----------|--------------------------|
| Swin-T  | ~28M      | Fast     | Good     | Quick experiments        |
| Swin-S  | ~50M      | Medium   | Better   | Balanced performance     |
| Swin-B  | ~88M      | Slow     | Great    | High accuracy needs      |
| Swin-L  | ~197M     | Slowest  | Best     | Maximum accuracy         |

### Image Size Trade-offs

| Size    | Speed    | Accuracy | Memory  |
|---------|----------|----------|---------|
| 224×224 | Fast     | Good     | Low     |
| 384×384 | Medium   | Better   | Medium  |
| 512×512 | Slow     | Best     | High    |

### Batch Size Guidelines

- **Small GPU (8GB)**: BATCH_SIZE = 8-16 (224px) or 4-8 (384px)
- **Medium GPU (16GB)**: BATCH_SIZE = 16-32 (224px) or 8-16 (384px)
- **Large GPU (24GB+)**: BATCH_SIZE = 32-64 (224px) or 16-32 (384px)

## 📈 Expected Training Time

On single GPU (NVIDIA RTX 3090):

| Config            | Time/Epoch | 100 Epochs |
|-------------------|------------|------------|
| Swin-T, 224, BS16 | ~2 min     | ~3.3 hrs   |
| Swin-T, 384, BS8  | ~4 min     | ~6.7 hrs   |
| Swin-S, 224, BS16 | ~3 min     | ~5 hrs     |
| Swin-B, 224, BS16 | ~5 min     | ~8.3 hrs   |

## 🎓 Training Tips

### For Best Accuracy
1. Use larger model (Swin-B or Swin-L)
2. Increase image size to 384 or 512
3. Train for more epochs (150-200)
4. Use lower learning rate (5e-5)
5. Enable class weights if imbalanced

### For Faster Training
1. Use Swin-T
2. Keep image size at 224
3. Enable mixed precision
4. Increase batch size
5. Cache images in memory (if RAM allows)

### For Better Generalization
1. Enable label smoothing (0.1-0.2)
2. Use strong augmentation
3. Apply weight decay (0.05)
4. Use dropout (if overfitting)
5. Monitor validation accuracy closely

## 🐛 Common Issues & Solutions

### 1. CUDA Out of Memory
**Solution**: Reduce BATCH_SIZE, use smaller IMG_SIZE, or use gradient accumulation

### 2. Training is too slow
**Solution**: Enable MIXED_PRECISION, increase NUM_WORKERS, or CACHE_IMAGES

### 3. Model not learning
**Solution**: Check learning rate, verify data loading, ensure correct loss function

### 4. Overfitting
**Solution**: Increase LABEL_SMOOTHING, reduce model size, add more augmentation

### 5. Underfitting
**Solution**: Increase model capacity, train longer, reduce regularization

## 📝 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir logs/
```

Metrics to watch:
- **Train/Val Loss**: Should decrease steadily
- **Train/Val Accuracy**: Should increase
- **Learning Rate**: Should follow warmup then decay schedule
- **Per-class Accuracy**: Check for class imbalance issues

### Console Output
```
Epoch 50/100 [Train]: 100%|████| 69/69 [01:45<00:00]
loss: 0.2341, acc: 91.23%, lr: 0.000087

[Epoch 50] Results:
  Train - Loss: 0.2341, Acc: 91.23%
  Val   - Loss: 0.2891, Acc: 89.45%
  Per-class accuracy:
    fire: 92.34%
    smoke: 88.12%
    neitherFireNorSmoke: 87.89%

✓ New best model! accuracy: 89.45%
```

## 🎯 Next Steps

1. **Start Training**: `python train.py`
2. **Monitor Progress**: Open TensorBoard
3. **Evaluate Results**: `python evaluate.py --checkpoint checkpoints/best_model.pth --split test`
4. **Fine-tune**: Adjust config.py based on results
5. **Inference**: Use inference.py for predictions

## 📚 File Structure After Training

```
swin-transformer-pytorch/
├── checkpoints/
│   ├── best_model.pth          # Best model by validation metric
│   └── checkpoint_epoch_*.pth  # Periodic checkpoints
├── logs/
│   └── run_20251110_143022/    # TensorBoard logs
├── results/
│   ├── evaluation_test/         # Test set evaluation
│   │   ├── confusion_matrix.png
│   │   ├── per_class_accuracy.png
│   │   ├── probability_distribution.png
│   │   └── metrics.txt
│   └── test_results.txt        # Final test results
└── ... (source files)
```

## 🔬 Understanding the Pipeline

### Data Flow
```
FASDD_RS Dataset
    ↓
FASSDDataset (data_utils.py)
    ↓ (transform + augment)
DataLoader (batch + shuffle)
    ↓
Swin Transformer Model
    ↓
Loss Calculation
    ↓
Backward Pass
    ↓
Optimizer Step
    ↓
Checkpoint Saving
```

### Key Components

1. **Dataset Loading** (`data_utils.py`)
   - Parses COCO/VOC/YOLO annotations
   - Loads TIFF images
   - Applies transformations
   - Returns (image, label) pairs

2. **Model** (`swin_transformer.py`)
   - Your pure Swin Transformer implementation
   - No modifications needed
   - Takes (B, C, H, W) images
   - Returns (B, num_classes) logits

3. **Training Loop** (`train.py`)
   - Orchestrates everything
   - Handles optimization
   - Manages checkpoints
   - Logs metrics

4. **Evaluation** (`evaluate.py`)
   - Computes metrics
   - Generates visualizations
   - Analyzes model performance

Everything is designed to work together seamlessly while using ONLY your Swin Transformer implementation!

---

**Ready to train? Run: `python train.py`** 🚀
