# TensorRT Conversion and Benchmarking Guide

## Overview

This guide explains how to convert your PyTorch model to TensorRT and benchmark it on your Jetson Orin device.

## Files

1. **`onnx2trt_benchmark.py`** - ONNX to TensorRT conversion and benchmarking (NEW)
2. **`benchmark.py`** - Updated CPU benchmarking with dataset testing
3. **`Step1_pth2trt.py`** - Original PTH to ONNX converter

## CUDA Kernel Error Issue

Your error `CUDA error: no kernel image is available for execution on the device` occurs because your PyTorch installation was compiled for different GPU architectures and doesn't include CUDA kernels for Jetson Orin (Compute Capability 8.7).

### Solution Options

#### Option 1: Install NVIDIA's JetPack PyTorch (Recommended)
```bash
# Uninstall current PyTorch
pip3 uninstall torch torchvision torchaudio

# Install NVIDIA's official PyTorch for Jetson
# Download from: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# Example for JetPack 5.1.2:
wget https://nvidia.box.com/shared/static/xxx.whl
pip3 install torch-*.whl
```

#### Option 2: Use NVIDIA's Docker Container
```bash
docker run -it --runtime nvidia --network host \
    nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
```

#### Option 3: Use CPU/TensorRT Only
Since TensorRT is optimized specifically for NVIDIA hardware, it's the best option for Jetson devices.

## Workflow

### Step 1: Test PyTorch Model (CPU only due to CUDA issue)

```bash
python test_pth.py
```

**What it does:**
- Loads your SwinTransformer model
- Tests on CPU (avoiding CUDA kernel error)
- Benchmarks performance
- Saves detailed report to `result/test_report_*.json` and `*.txt`

### Step 2: Convert ONNX to TensorRT

```bash
python onnx2trt_benchmark.py
```

**What it does:**
1. Converts `model.onnx` → `model.trt` (or `model_FP16.trt` if FP16 supported)
2. Automatically detects and uses FP16 if your GPU supports it
3. Benchmarks TensorRT engine performance
4. Tests on images from your `dataset/` folder
5. Saves comprehensive results to `result/trt_report_*.json` and `*.txt`

**Output files:**
- `model.trt` or `model_FP16.trt` - TensorRT engine
- `result/trt_report_YYYYMMDD_HHMMSS.json` - JSON results
- `result/trt_report_YYYYMMDD_HHMMSS.txt` - Human-readable report

### Step 3: CPU Benchmark with Dataset Testing

```bash
python benchmark.py
```

**What it does:**
- Loads PyTorch model
- Benchmarks on CPU
- Tests on multiple images from dataset
- Saves results to `result/cpu_inference_results_*.json`

## Installation Requirements

### Core Dependencies
```bash
# NumPy (must be <2.0 for PyCUDA compatibility)
pip3 install "numpy<2.0"

# PyTorch (use NVIDIA's JetPack version)
pip3 install torch torchvision

# TensorRT (should be pre-installed on JetPack)
# If not: pip3 install tensorrt

# PyCUDA for TensorRT inference
pip3 install pycuda

# ONNX tools for model conversion
pip3 install onnx onnx-simplifier

# Other dependencies
pip3 install opencv-python pillow
```

### Verify TensorRT Installation
```bash
python3 -c "import tensorrt as trt; print(f'TensorRT version: {trt.__version__}')"
```

## Configuration

Edit the configuration section in each script:

### `onnx2trt_benchmark.py`
```python
ONNX_PATH = "/home/dronex/Documents/TUBITAK-1002/model.onnx"
TRT_PATH = "/home/dronex/Documents/TUBITAK-1002/model.trt"
DATASET_FOLDER = "/home/dronex/Documents/TUBITAK-1002/dataset"
OUTPUT_DIR = "result"
CLASS_NAMES = ['fire', 'neither', 'smoke']
```

### `benchmark.py`
```python
MODEL_PATH = "/home/dronex/Documents/TUBITAK-1002/Models/best_model_base.pth"
BATCH_SIZE = 8  # Adjust based on your GPU memory
DATASET_FOLDER = "/home/dronex/Documents/TUBITAK-1002/dataset"
CLASS_NAMES = ['fire', 'neither', 'smoke']
```

## Expected Performance

### Typical Performance on Jetson Orin Nano

| Method | Inference Time | Throughput |
|--------|---------------|------------|
| PyTorch CPU | ~500-1000 ms | 1-2 FPS |
| PyTorch CUDA | ~50-100 ms | 10-20 FPS |
| TensorRT FP32 | ~20-40 ms | 25-50 FPS |
| TensorRT FP16 | ~10-20 ms | 50-100 FPS |

*Note: Actual performance depends on model size, batch size, and system configuration*

## Understanding the Reports

### TensorRT Report Structure

```
result/trt_report_20260115_210055.json
├── timestamp
├── conversion
│   ├── onnx_path
│   ├── trt_path
│   └── fp16_mode
├── benchmark
│   ├── avg_time_ms
│   ├── throughput_fps
│   └── ...
└── dataset_test
    ├── images (per-image results)
    └── summary
        ├── total_images
        ├── avg_inference_time_ms
        └── class_distribution
```

## Troubleshooting

### Issue: "Failed to parse ONNX file" or "UNSUPPORTED_NODE_ATTR: ScatterND reduction"
**Problem:** Your SwinTransformer model uses ScatterND operations with a "reduction" attribute that TensorRT 10.3 doesn't support. This is a known limitation for complex vision transformer models.

**Root Cause:** The model's relative position embedding uses ScatterND with reduction, which was added in ONNX opset 16+ but isn't fully supported in TensorRT yet.

**Solution Options:**

**Option 1: Use CPU Inference (Recommended for now)**
```bash
# Your model works perfectly on CPU
python benchmark.py

# This will test on your dataset and save results
```

**Option 2: Modify Model to Disable Relative Position Embedding**  
Create a new checkpoint with `relative_pos_embedding=False`:
```python
# In your training script, when creating the model:
model = SwinTransformer(
    ...
    relative_pos_embedding=False  # Disable this feature for TensorRT
)
# Retrain or create a new checkpoint
```

**Option 3: Use torch-tensorrt (if available on your PyTorch)**
```bash
pip3 install torch-tensorrt

# Then use benchmark.py which already has torch-tensorrt support
python benchmark.py
```

**Option 4: Wait for TensorRT Update**  
NVIDIA is actively working on supporting more ONNX operations. Check for TensorRT updates:
```bash
# On JetPack, TensorRT is updated with JetPack updates
sudo apt update && sudo apt upgrade nvidia-jetpack
```

### Current Recommendation
Given the CUDA kernel error with PyTorch and the TensorRT  ONNX compatibility issue:
1. Use **CPU inference** with the optimized `benchmark.py` script
2. Consider retraining with `relative_pos_embedding=False` if TensorRT is critical
3. Or use a different model architecture that's more TensorRT-friendly (ResNet, EfficientNet, etc.)

### Issue: "Out of memory" during TensorRT build
**Solution:** Reduce max_workspace_size in onnx2trt_benchmark.py:
```python
max_workspace_size=1<<28  # 256MB instead of 1GB
```

### Issue: "PyCUDA not installed"
**Solution:**
```bash
pip3 install pycuda
```

### Issue: TensorRT not found
**Solution:**
```bash
# On Jetson with JetPack:
sudo apt-get install python3-libnvinfer-dev

# Or add to Python path:
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
```

## Best Practices

1. **Always test on CPU first** to verify model loads correctly
2. **Use FP16 mode** for TensorRT on Jetson - it's much faster and barely affects accuracy
3. **Start with small batch sizes** (1-8) to avoid memory issues
4. **Keep workspace size reasonable** (1GB max on Jetson Orin Nano)
5. **Test on sample images** before processing entire dataset

## Performance Optimization Tips

### For TensorRT Conversion
- Enable FP16 mode (usually 2x faster with minimal accuracy loss)
- Adjust workspace size based on available GPU memory
- Use static input shapes when possible

### For Inference
- Batch multiple images together
- Preprocess images in parallel
- Use GPU for preprocessing when possible

## Next Steps

1. Run `python onnx2trt_benchmark.py` to convert and benchmark
2. Check `result/` folder for detailed reports
3. If TensorRT works well, integrate it into your production pipeline
4. Consider creating a TensorRT inference server for deployment

## Additional Resources

- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Jetson PyTorch Wheels](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
- [PyCUDA Documentation](https://documen.tician.de/pycuda/)
