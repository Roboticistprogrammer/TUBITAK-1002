# Project Status Report - Swin Transformer Deployment

**Generated:** January 15, 2026  
**Model:** Swin Transformer (86.7M parameters, 330MB)  
**Target:** Jetson Orin Nano  

---

## Current Situation

### ✅ What Works

1. **PyTorch Model on CPU** - Fully functional
   - Model loads correctly
   - Inference works
   - Detailed testing with `test_pth.py`
   - Benchmarking with `benchmark.py`
   - Dataset testing implemented

2. **ONNX Export** - Successfully created
   - Original: `model.onnx` (333.6 MB)
   - Simplified: `model_simplified.onnx` (330.9 MB)

### ❌ What Doesn't Work

1. **PyTorch CUDA Inference**
   - **Error:** `CUDA error: no kernel image is available for execution on the device`
   - **Reason:** Your PyTorch 2.9.0+cu126 wasn't compiled for Jetson Orin's GPU architecture (Compute Capability 8.7)
   - **Impact:** Cannot use GPU with PyTorch directly

2. **TensorRT Conversion from ONNX**
   - **Error:** `UNSUPPORTED_NODE_ATTR: ScatterND with reduction attribute not supported`
   - **Reason:** SwinTransformer uses relative position embeddings that generate ScatterND ops with reduction
   - **TensorRT Version:** 10.3.0 doesn't support this operation yet
   - **Impact:** Cannot convert to TensorRT engine

---

## Why These Issues Occur

### CUDA Kernel Error
PyTorch binaries from PyPI are compiled for specific GPU architectures. Your installation doesn't include kernels for:
- Compute Capability 8.7 (Jetson Orin)

You need NVIDIA's custom-built PyTorch for Jetson (from JetPack).

### TensorRT ScatterND Issue
The Swin Transformer architecture uses:
```python
relative_pos_embedding=True  # This causes ScatterND with reduction
```

This operation:
- Is valid in ONNX (opset 16+)
- Works in ONNX Runtime
- **Is NOT supported in TensorRT 10.3**

---

## Available Solutions

### Solution 1: CPU Inference (Ready Now) ✅

**Best for:** Development, testing, low-throughput scenarios

```bash
# Test and benchmark
python test_pth.py

# Run on dataset
python benchmark.py
```

**Performance:** ~500-1000ms per image  
**Advantages:**
- Works immediately
- No compatibility issues
- Full model functionality
- Detailed reports in `result/` folder

---

### Solution 2: Install JetPack PyTorch (Enables GPU)

**Best for:** Maximum performance with PyTorch

**Steps:**
1. Visit: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
2. Download JetPack-compatible PyTorch wheel
3. Install:
```bash
pip3 uninstall torch torchvision
pip3 install torch-<version>-cp310-cp310-linux_aarch64.whl
```

**Expected Performance:** ~50-100ms per image  
**Note:** Still cannot use TensorRT due to ScatterND issue

---

### Solution 3: Retrain Without Relative Position Embedding

**Best for:** TensorRT deployment

**Modify model creation:**
```python
model = SwinTransformer(
    hidden_dim=128,
    layers=(2, 2, 18, 2),
    heads=(4, 8, 16, 32),
    num_classes=3,
    window_size=7,
    relative_pos_embedding=False,  # ← Change this
    ...
)
```

**Then:**
1. Retrain model
2. Export to ONNX
3. Convert to TensorRT
4. Get 10-20ms inference time

**Accuracy Impact:** Minor (usually <1% degradation)

---

### Solution 4: Use Different Architecture

**Best for:** Fastest time-to-deployment with TensorRT

**TensorRT-friendly architectures:**
- ResNet-50/101
- EfficientNet-B0 to B7
- MobileNet V2/V3
- RegNet
- ConvNeXt

These all convert smoothly to TensorRT.

---

## Recommended Path Forward

### Immediate Term (This Week)
1. **Use CPU inference** with the optimized scripts provided
2. Process your dataset with `benchmark.py`
3. Collect baseline metrics

### Short Term (1-2 Weeks)
1. **Install JetPack PyTorch** for GPU acceleration
   - Download from NVIDIA forums
   - Test with same scripts
   - Compare GPU vs CPU performance

### Long Term (If TensorRT is Critical)

**Option A:** Wait for TensorRT update
- Monitor NVIDIA's TensorRT releases
- ScatterND with reduction support may be added

**Option B:** Retrain without relative position embedding
- Modify model config
- Retrain on your dataset
- Export and deploy to TensorRT
- Achieve <20ms inference

**Option C:** Switch to TensorRT-optimized architecture
- Choose from ResNet/EfficientNet/MobileNet
- Retrain
- Full TensorRT support

---

## Files Created for You

| File | Purpose | Status |
|------|---------|--------|
| `test_pth.py` | Model testing & reporting | ✅ Working |
| `benchmark.py` | CPU benchmarking & dataset testing | ✅ Working |
| `onnx2trt_benchmark.py` | ONNX→TensorRT conversion | ❌ Blocked by ScatterND |
| `TENSORRT_GUIDE.md` | Comprehensive guide | 📖 Reference |
| `model_simplified.onnx` | Optimized ONNX model | ✅ Created |
| `result/*.json` | Test reports | ✅ Generated |

---

## Performance Summary

| Method | Status | Inference Time | Notes |
|--------|--------|---------------|-------|
| PyTorch CPU | ✅ Working | ~500-1000ms | Available now |
| PyTorch GPU | ❌ CUDA Error | N/A | Need JetPack PyTorch |
| TensorRT | ❌ ScatterND | N/A | Need model modification |

---

## Next Steps

1. **Run the working scripts:**
```bash
# Full model test
python test_pth.py

# Dataset benchmark
python benchmark.py
```

2. **Review results:**
```bash
ls -lh result/
# Check the generated JSON and TXT files
```

3. **Decide on deployment strategy:**
   - CPU-only? → Continue with current setup
   - Need GPU? → Install JetPack PyTorch
   - Need TensorRT? → Retrain model or change architecture

---

## Questions?

If you need help with:
- Installing JetPack PyTorch
- Retraining without relative position embedding
- Switching to a different architecture
- Optimizing CPU performance further

Let me know and I can provide specific guidance!

---

**Bottom Line:** Your model works great on CPU right now. For GPU/TensorRT, you need either different PyTorch (for GPU) or model modifications (for TensorRT).
