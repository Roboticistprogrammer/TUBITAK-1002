from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch


class InferenceRunner(ABC):
    @abstractmethod
    def infer(self, images: torch.Tensor) -> np.ndarray:
        """Return float32 logits with shape [batch, classes]."""


class PyTorchRunner(InferenceRunner):
    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device

    def infer(self, images: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            outputs = self.model(images.to(self.device, non_blocking=True))
            logits = getattr(outputs, "logits", outputs)
        return logits.float().cpu().numpy()


class OnnxRuntimeRunner(InferenceRunner):
    def __init__(self, model_path: Path, use_cuda: bool = True) -> None:
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = []
        if use_cuda and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.providers = self.session.get_providers()

    def infer(self, images: torch.Tensor) -> np.ndarray:
        batch = np.ascontiguousarray(images.cpu().numpy(), dtype=np.float32)
        return self.session.run([self.output_name], {self.input_name: batch})[0].astype(
            np.float32, copy=False
        )


class TensorRTRunner(InferenceRunner):
    """TensorRT 10+ runner backed by torch CUDA allocations."""

    def __init__(self, engine_path: Path, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError("TensorRT evaluation requires a CUDA device.")
        import tensorrt as trt

        self.trt = trt
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(
                "TensorRT could not deserialize the engine. Rebuild it in this exact container and on this target GPU."
            )
        self.context = self.engine.create_execution_context()
        self.input_name = next(
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.INPUT
        )
        self.output_name = next(
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.OUTPUT
        )

    def infer(self, images: torch.Tensor) -> np.ndarray:
        images = images.to(self.device, dtype=torch.float32, non_blocking=True).contiguous()
        self.context.set_input_shape(self.input_name, tuple(images.shape))
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        output = torch.empty(output_shape, dtype=torch.float32, device=self.device)
        self.context.set_tensor_address(self.input_name, images.data_ptr())
        self.context.set_tensor_address(self.output_name, output.data_ptr())
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT inference execution failed.")
        stream.synchronize()
        return output.cpu().numpy()
