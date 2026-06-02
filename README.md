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
- Fix the model path reference (currently has syntax error: `./Models/best_model.pth`)
- Specify correct input tensor shape for Swin Transformer
- Set appropriate model initialization parameters

### Issues to Fix in `Step1_pth2trt.py`:

```python
# Current (incorrect):
model = ./Models/best_model.pth

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

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Use CPU: `export CUDA_VISIBLE_DEVICES=""` then run script, or reduce batch size |
| Model not found | Check path: `Models/best_model_base.pth` |
| ONNX export fails | Verify model input/output shapes, try CPU-only conversion |
| TensorRT build fails | Check CUDA/TensorRT version compatibility, verify model conversion first |
| FP16 not supported | Falls back to default precision (FP32) |

docker run --runtime=nvidia --rm -p 8000:8000 -p 8001:8001 -p 8002:8002   -v ${PWD}/model_repository:/models   nvcr.io/nvidia/deepstream:7.1-triton-multiarch   tritonserver --model-repository=/models