# Swin Transformer Model Testing Guide

## Overview
This guide covers testing and converting your Swin Transformer model (`best_model_base.pth`) for deployment with TensorRT optimization.

## Prerequisites

```bash
pip install torch torchvision
pip install tensorrt
pip install onnx onnxruntime
pip install opencv-python numpy
```

## Model Files

- **Model Weight**: `Models/best_model_base.pth`
- **Test Scripts**:
  - `test_pth.py` - Test PyTorch model directly
  - `test_trt.py` - Test TensorRT converted model
  - `Step1_pth2trt.py` - Convert .pth → .onnx → .trt

## Step 1: Test PyTorch Model

**First**, identify your model architecture by running:

```bash
python check_model_architecture.py
```

This will show you the model structure and save details to `model_analysis.json`.

Then run `test_pth.py` to verify your Swin Transformer model works correctly:

```bash
python test_pth.py
```

**Note**: If you get "model_state_dict mismatch" errors, see the **Model Architecture Notes** section below.

## Step 2: Convert Model to ONNX

Use the conversion script to convert your PyTorch model to ONNX format:

```bash
python Step1_pth2trt.py
```

**Note**: Before running, update `Step1_pth2trt.py`:
- Fix the model path reference (currently has syntax error: `./Models/best_model_base.pth`)
- Specify correct input tensor shape for Swin Transformer
- Set appropriate model initialization parameters

### Issues to Fix in `Step1_pth2trt.py`:

```python
# Current (incorrect):
model = ./Models/best_model_base.pth

# Should be:
model = YourSwinTransformerClass()  # Define your model architecture
model.load_state_dict(torch.load("Models/best_model_base.pth"))
```

## Step 3: Test TensorRT Model

After conversion, test the optimized TensorRT model:

```bash
python test_trt.py
```

## Workflow Summary

```
best_model_base.pth (PyTorch)
    ↓
    Step1_pth2trt.py
    ↓
mnist_model.onnx
    ↓
    Step1_pth2trt.py (Onnx2Trt)
    ↓
mnist_model_FP16.trt (or mnist_model.trt)
    ↓
test_trt.py
```

## Dataset Testing

Once model is validated, test with actual fire detection dataset:

```bash
# Setup datasets
python download_dataset.py --setup

# Extract datasets
python download_dataset.py --extract

# Preprocess images
# Use Preprocess.ipynb notebook
```

## Model Architecture Notes

**IMPORTANT**: Your checkpoint contains a **custom Swin Transformer** implementation (with `stage1`, `stage2`, etc.), not the standard torchvision `swin_b` model.

To load your model correctly, you need to:

1. **Find or define your custom model class** that matches the checkpoint architecture
2. Update the test scripts to use your custom model instead of `swin_b()`

### Checking Your Model Architecture

Run this diagnostic to see the expected model structure:

```python
import torch
checkpoint = torch.load('Models/best_model_base.pth', map_location='cpu')
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint
print("Model keys:")
for key in list(state_dict.keys())[:10]:
    print(f"  {key}")
```

Expected output will show layer names like `stage1.patch_partition.linear.weight` instead of `features.0.0.weight`

### Solution

Your `.pth` file likely comes from a custom training script. You need to:

1. **Locate your model definition** - Find where this architecture is defined
2. **Update scripts** - Replace `swin_b()` with your custom model class
3. **Example fix**:

```python
# Instead of:
# model = swin_b(weights=None)

# Use your custom model:
# from your_module import CustomSwinTransformer
# model = CustomSwinTransformer(...)
model.load_state_dict(state_dict)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Use CPU: `export CUDA_VISIBLE_DEVICES=""` then run script, or reduce batch size |
| Model not found | Check path: `Models/best_model_base.pth` |
| ONNX export fails | Verify model input/output shapes, try CPU-only conversion |
| TensorRT build fails | Check CUDA/TensorRT version compatibility, verify model conversion first |
| FP16 not supported | Falls back to default precision (FP32) |

### CUDA Out of Memory Solutions

**Option 1: Run on CPU (Slower but safe)**
```bash
export CUDA_VISIBLE_DEVICES=""
python test_pth.py
```

**Option 2: Reduce GPU memory usage**
- Clear GPU cache: `nvidia-smi` to see GPU usage
- Close other GPU applications
- Reduce batch size in the script
- Use mixed precision (FP16 during inference)

**Option 3: Check GPU memory**
```bash
nvidia-smi
# or
python -c "import torch; print(torch.cuda.get_device_properties(0))"
```

**Option 4: Monitor GPU during execution**
```bash
watch -n 1 nvidia-smi
```

## Performance Comparison

After testing all three formats:
- **PyTorch**: Baseline accuracy/speed
- **ONNX**: Cross-platform compatibility
- **TensorRT**: Optimized inference speed (2-10x faster on GPU)

## Next Steps

1. ✅ Test PyTorch model (`test_pth.py`)
2. ✅ Convert to ONNX + TensorRT (`Step1_pth2trt.py`)
3. ✅ Test TensorRT model (`test_trt.py`)
4. ✅ Benchmark all three formats
5. Deploy TensorRT model for production

## References

- [check_model_architecture.py](check_model_architecture.py) - Model diagnostic script
- [Step1_pth2trt.py](Step1_pth2trt.py) - Conversion script
- [Models/best_model_base.pth](Models/best_model_base.pth) - Custom Swin Transformer weights
- [test_pth.py](test_pth.py) - PyTorch testing
- [test_trt.py](test_trt.py) - TensorRT testing
- [Preprocess.ipynb](Preprocess.ipynb) - Dataset preprocessing