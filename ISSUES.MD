# Known Issues & Workarounds

---

## 1. Teacher `.pt` files are raw state_dict checkpoints — not Ultralytics models

**Symptom**
```
This .pt appears to be a raw state_dict checkpoint (OrderedDict),
not an Ultralytics YOLO model.
```
**Root cause**  
The teacher models were trained with a custom SwinV2 trainer, not Ultralytics. The `.pt` files hold a plain `OrderedDict` (weight tensors only), not a wrapped `YOLO` / `nn.Module` object.

**Current workaround**  
- Teachers are excluded from the pipeline (`model_config.json` only has student models).  
- `Benchmark_Engine.py` falls back to `benchmark_model_raw_pt()`, which reports **file size only** because no callable module can be extracted from the checkpoint.

**Proper fix**  
Export each teacher to ONNX using the original training code:
```python
# inside original training codebase
torch.onnx.export(model, dummy_input, "teacher_cv.onnx", opset_version=17)
```
Then add the `.onnx` files to `model_config.json` for full benchmarking.

---

## 2. TensorRT engine version mismatch

**Symptom**
```
ERROR ❌ TensorRT model exported with a different version than expected
AttributeError: 'NoneType' object has no attribute 'create_execution_context'
```
**Root cause**  
`model.engine` was built with a different TensorRT version than what is currently installed (TRT 11.0.0.114). TRT engines are **not portable across versions**.

**Current workaround**  
`_benchmark_tensorrt()` catches the `None` engine and reports file size only with a clear version message.

**Proper fix**  
Re-export the engine from the ONNX model using the installed TRT version:
```bash
yolo export model=Models/students/last.pt format=engine imgsz=224 device=0
# or from the onnx directly:
trtexec --onnx=Models/model.onnx --saveEngine=Models/model.engine --inputIOFormats=fp16:chw --fp16
```
Then re-run the pipeline — `_benchmark_tensorrt()` will run the full latency/FPS/power loop.

---

## 3. ONNX model input size is 224×224, not 640×640

**Symptom**
```
Got invalid dimensions for input: index 2 Got: 640 Expected: 224
```
**Root cause**  
The student model was exported at `imgsz=224`. Ultralytics defaults to 640 and crashes.

**Current workaround**  
- Added `"imgsz": 224` to `model_config.json` for ONNX and engine entries.
- `Benchmark_Engine.py` accepts `--imgsz` and reads it from config.
- `_benchmark_onnxruntime()` auto-reads the declared shape from the ONNX graph itself as a safety net.

**Note**  
If you re-export, keep `imgsz=224` consistent across `.onnx` and `.engine`.

---

## 4. Wrong Python interpreter selected on WSL (Windows `.exe` picked up)

**Symptom**
```
Using Python interpreter: /mnt/.../TUBITAK-1002/.venv/Scripts/python.exe
Error: E:\...\best.pt — Windows path used on Linux subprocess
```
**Root cause**  
`detect_python_executable()` was checking `.venv/Scripts/python.exe` even on Linux/WSL, picking up a Windows venv that converts paths to `E:\...`.

**Fix applied**  
`run_full_pipeline.py` now skips the `Scripts/python.exe` candidate when `platform.system() == "Linux"`.

---

## 5. `ultralytics` / `onnx` / `onnxruntime-gpu` / `tensorrt` / `pycuda` not installed

**Symptom**  
`ModuleNotFoundError` for any of the above when the pipeline runs subprocesses.

**Root cause**  
System Python (`/usr/bin/python3`) was missing these packages. Ultralytics' auto-install mechanism fails with permission errors on the system Python.

**Fix applied — install once manually:**
```bash
pip install ultralytics onnx onnxruntime-gpu tensorrt-cu12 pycuda nvidia-ml-py \
    --break-system-packages
```

---

## 6. NumPy 2.x vs system Matplotlib 1.x ABI conflict

**Symptom**
```
A module compiled using NumPy 1.x cannot be run in NumPy 2.4.6
ImportError: numpy.core.multiarray failed to import
```
**Root cause**  
Pip installed NumPy 2.4.6 but the system `matplotlib` package was compiled against NumPy 1.x.

**Fix applied:**
```bash
pip install "matplotlib>=3.9" --break-system-packages
```
`matplotlib 3.9+` has full NumPy 2.x support and overrides the system package in user site-packages.

---

## 7. `pynvml` FutureWarning from inside PyTorch

**Symptom**
```
FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead.
  import pynvml  # inside torch/cuda/__init__.py
```
**Root cause**  
PyTorch itself imports the old `pynvml` package internally. This cannot be silenced from outside torch.

**Workaround**  
This warning is **harmless** — power monitoring in `Benchmark_Engine.py` uses `nvidia-ml-py` which exposes the same `pynvml` API. The warning comes from torch's own code and can be ignored. To suppress it in your own imports, `Benchmark_Engine.py` wraps its import in `warnings.catch_warnings()`.

---

## 8. Student `.pt` — raw state_dict with unknown architecture

**Symptom**  
`benchmark_model_raw_pt()` cannot find a callable module:
```
[INFO] Checkpoint is a raw state_dict — reporting file size only.
```
**Root cause**  
`Models/students/last.pt` is an `OrderedDict` of weights without the model class attached. `torch.load()` returns a plain dict, not an `nn.Module`.

**Fix**  
Use the original training code to wrap the model and export to ONNX:
```python
model = SwinV2Base(...)          # instantiate architecture
state = torch.load("last.pt")["model"]   # or whatever key holds the weights
model.load_state_dict(state)
torch.onnx.export(model, dummy, "student.onnx", opset_version=17, input_names=["input"])
```
Replace `Models/model.onnx` with this export. The ONNX path already works end-to-end.

---

## Quick reference — environment setup from scratch

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126 \
    --break-system-packages

pip install ultralytics onnx onnxruntime-gpu \
    tensorrt-cu12 pycuda \
    nvidia-ml-py "matplotlib>=3.9" \
    --break-system-packages
```
