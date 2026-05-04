To convert a .pt PyTorch model to a .engine TensorRT file within a Docker container, the standard workflow involves first exporting the model to ONNX format and then using NVIDIA's trtexec tool to build the engine.1. Launch a TensorRT-capable ContainerStart by running an official NVIDIA PyTorch container that includes TensorRT. Mount your local model directory to the /workspace folder inside the container.bashdocker run --rm -it --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:24.03-py3 bash
Use code with caution.--gpus all: Grants the container access to your host GPU.-v $(pwd):/workspace: Maps your current directory (containing your .pt file) into the container.2. Export the Model to ONNXInside the container, you must convert the .pt file to an intermediate ONNX format.For YOLOv8/v10/v11 (Ultralytics): Use the Ultralytics CLI for a direct conversion.bashpip install ultralytics
yolo export model=your_model.pt format=onnx
Use code with caution.For Custom Models: Use a Python script with torch.onnx.export.pythonimport torch
model = torch.load('your_model.pt')
dummy_input = torch.randn(1, 3, 224, 224).cuda()
torch.onnx.export(model, dummy_input, "your_model.onnx")
Use code with caution.3. Generate the Engine FileUse the trtexec tool (pre-installed in NVIDIA containers) to build the hardware-specific .engine file from your ONNX model.bashtrtexec --onnx=your_model.onnx \
        --saveEngine=your_model.engine \
        --fp16
Use code with caution.--fp16: (Optional) Enables half-precision for significant speedups.--saveEngine: Specifies the output path for the finalized engine file.