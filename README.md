# Swin Transformer for Fire and Smoke Detection

This repository contains a PyTorch implementation of Swin Transformer for fire and smoke detection on the FASDD (Fire and Smoke Detection Dataset) family of datasets.

## 🔥 Current Branch: Swin-B (Base) Architecture

This branch is configured to train using the **Swin Transformer Base** architecture with the following specifications:

- **Model Variant**: Swin-B (Base)
- **Hidden Dimension**: 128
- **Layers**: (2, 2, 18, 2)
- **Attention Heads**: (4, 8, 16, 32)
- **Window Size**: 7
- **Input Image Size**: 224x224
- **Parameters**: ~88M

## 📊 Datasets

The training pipeline supports multi-dataset training with three FASDD variants:
- **FASDD_RS**: Remote Sensing dataset
- **FASDD_CV**: Computer Vision dataset  
- **FASDD_UAV**: UAV (Unmanned Aerial Vehicle) dataset

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

The configuration is set in [config.py](config.py) with:
- Batch size: 16
- Learning rate: 1e-4
- Epochs: 100
- Optimizer: AdamW
- Scheduler: Cosine with warmup

### Evaluation

```bash
python evaluate.py
```

### Inference

```bash
python inference.py --image_path /path/to/image.jpg
```

## 📁 Project Structure

```
swin-transform-pytorch/
├── config.py                 # Training configuration (Swin-B settings)
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── inference.py              # Inference script
├── test_datasets.py          # Dataset testing utilities
├── swin_transformer_pytorch/ # Model implementation
│   ├── models/
│   │   └── swin_transformer.py
│   └── data_utils.py
├── docs/                     # Documentation
│   ├── README_TRAINING.md
│   ├── TRAINING_GUIDE.md
│   └── COMBINED_DATASET_README.md
└── results/                  # Training results and metrics
```

## 🎯 Model Architecture Details

The Swin-B configuration uses:
- 4 stages with progressive downsampling
- Window-based self-attention with shifted windows
- Relative position bias
- Layer normalization and GELU activation
- Patch embedding with 4x4 patch size

See [config.py](config.py) for full configuration details.

## 📝 Configuration

Key configuration parameters in [`Config`](config.py) class:

- **Model**: [`MODEL_VARIANT`](config.py) = 'base'
- **Datasets**: [`USE_MULTI_DATASETS`](config.py) = True
- **Classes**: 3 (fire, smoke, neitherFireNorSmoke)
- **Mixed Precision**: Enabled
- **Early Stopping**: 15 epochs patience

To modify settings, edit [config.py](config.py) directly.

## 📚 Documentation

- [Training Guide](docs/TRAINING_GUIDE.md) - Detailed training instructions
- [Training README](docs/README_TRAINING.md) - Training configuration details
- [Combined Dataset README](docs/COMBINED_DATASET_README.md) - Multi-dataset setup

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

See [requirements.txt](requirements.txt) for full dependencies.

## 📈 Results

Training results and metrics are saved to:
- Checkpoints: `checkpoints/`
- Logs: `logs/`
- Results: `results/`

## 🌟 Features

- ✅ Multi-dataset training support
- ✅ Mixed precision training (AMP)
- ✅ Cosine learning rate scheduling with warmup
- ✅ Early stopping
- ✅ TensorBoard logging
- ✅ Best model checkpointing
- ✅ COCO annotation format support

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

Based on the Swin Transformer architecture from Microsoft Research.