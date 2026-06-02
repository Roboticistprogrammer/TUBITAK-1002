"""
Optimized ONNX to TensorRT Converter and Benchmark Tool
Based on: https://github.com/ycchen218/Pytorch-to-TensorRT-example

This script:
1. Converts ONNX model to TensorRT engine (.trt file)
2. Benchmarks TensorRT inference performance
3. Tests on images from dataset folder
4. Saves detailed results to 'result' folder
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from PIL import Image
import cv2


class TensorRTConverter:
    """Convert ONNX model to TensorRT engine with optimization"""
    
    def __init__(self, onnx_path, trt_path=None, fp16_mode=True, max_workspace_size=1<<30):
        """
        Args:
            onnx_path: Path to ONNX model file
            trt_path: Path to save TensorRT engine (if None, uses onnx_path with .trt extension)
            fp16_mode: Enable FP16 precision if supported
            max_workspace_size: Maximum GPU memory for TensorRT builder (default 1GB)
        """
        self.onnx_path = Path(onnx_path)
        if trt_path is None:
            self.trt_path = self.onnx_path.with_suffix('.trt')
        else:
            self.trt_path = Path(trt_path)
        
        self.fp16_mode = fp16_mode
        self.max_workspace_size = max_workspace_size
        self.logger = trt.Logger(trt.Logger.INFO)
        
    def build_engine(self):
        """Build TensorRT engine from ONNX model"""
        print(f"{'='*70}")
        print(f"Building TensorRT Engine")
        print(f"{'='*70}")
        print(f"ONNX Model: {self.onnx_path}")
        print(f"Output: {self.trt_path}")
        print(f"FP16 Mode: {self.fp16_mode}")
        print(f"Max Workspace: {self.max_workspace_size / (1024**3):.2f} GB\n")
        
        # Create builder and network
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        
        with trt.Builder(self.logger) as builder, \
             builder.create_network(EXPLICIT_BATCH) as network, \
             trt.OnnxParser(network, self.logger) as parser:
            
            # Configure builder
            config = builder.create_builder_config()
            
            # Set memory pool limit (updated API for newer TensorRT versions)
            if hasattr(config, 'set_memory_pool_limit'):
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.max_workspace_size)
            elif hasattr(config, 'max_workspace_size'):
                config.max_workspace_size = self.max_workspace_size
            
            # Enable FP16 if supported
            if self.fp16_mode and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("✓ FP16 mode enabled")
                self.trt_path = self.trt_path.with_name(
                    self.trt_path.stem + "_FP16" + self.trt_path.suffix
                )
            else:
                if self.fp16_mode:
                    print("⚠ FP16 not supported on this platform, using FP32")
                else:
                    print("✓ Using FP32 precision")
            
            # Parse ONNX model
            print("\nParsing ONNX model...")
            with open(self.onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    print('✗ ERROR: Failed to parse ONNX file')
                    for error in range(parser.num_errors):
                        print(f"  Error {error}: {parser.get_error(error)}")
                    return None
            
            print("✓ ONNX parsing completed")
            
            # Get input/output info
            print(f"\nNetwork Information:")
            print(f"  Inputs: {network.num_inputs}")
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                print(f"    [{i}] {input_tensor.name}: {input_tensor.shape}")
            
            print(f"  Outputs: {network.num_outputs}")
            for i in range(network.num_outputs):
                output_tensor = network.get_output(i)
                print(f"    [{i}] {output_tensor.name}: {output_tensor.shape}")
            
            # Build engine
            print("\nBuilding TensorRT engine (this may take several minutes)...")
            start_time = time.time()
            
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                print("✗ ERROR: Failed to build TensorRT engine")
                return None
            
            build_time = time.time() - start_time
            print(f"✓ Engine built successfully in {build_time:.2f}s")
            
            # Save engine
            with open(self.trt_path, 'wb') as f:
                f.write(serialized_engine)
            
            print(f"✓ Engine saved to {self.trt_path}")
            print(f"  Engine size: {self.trt_path.stat().st_size / (1024**2):.2f} MB\n")
            
            return self.trt_path


class TensorRTInference:
    """TensorRT inference engine wrapper"""
    
    def __init__(self, trt_engine_path):
        """
        Args:
            trt_engine_path: Path to TensorRT engine file
        """
        self.trt_engine_path = Path(trt_engine_path)
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        print(f"Loading TensorRT engine from {self.trt_engine_path}")
        with open(self.trt_engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
        self.context = self.engine.create_execution_context()
        
        # Get input/output info
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        
        print(f"✓ Engine loaded successfully")
        print(f"  Input: {self.input_name} {list(self.input_shape)}")
        print(f"  Output: {self.output_name} {list(self.output_shape)}\n")
        
        # Allocate buffers
        self.input_size = trt.volume(self.input_shape)
        self.output_size = trt.volume(self.output_shape)
        
        self.d_input = cuda.mem_alloc(self.input_size * np.dtype(np.float32).itemsize)
        self.d_output = cuda.mem_alloc(self.output_size * np.dtype(np.float32).itemsize)
        
        self.stream = cuda.Stream()
        
    def infer(self, input_data):
        """
        Run inference on input data
        
        Args:
            input_data: numpy array with shape matching input_shape
            
        Returns:
            Output numpy array
        """
        # Ensure correct shape and type
        if input_data.shape != tuple(self.input_shape):
            raise ValueError(f"Input shape {input_data.shape} doesn't match expected {self.input_shape}")
        
        input_data = np.ascontiguousarray(input_data.astype(np.float32))
        
        # Copy input to device
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        
        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        
        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Copy output to host
        output_data = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output_data, self.d_output, self.stream)
        self.stream.synchronize()
        
        return output_data
    
    def __del__(self):
        """Cleanup GPU memory"""
        if hasattr(self, 'd_input'):
            self.d_input.free()
        if hasattr(self, 'd_output'):
            self.d_output.free()


def preprocess_image(image_path, input_shape=(224, 224)):
    """
    Preprocess image for model input
    
    Args:
        image_path: Path to image file
        input_shape: Target image size (height, width)
        
    Returns:
        Preprocessed numpy array
    """
    img = cv2.imread(str(image_path))
    if img is None:
        img = np.array(Image.open(image_path).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Resize
    img = cv2.resize(img, input_shape)
    
    # Convert to RGB and normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    # Convert to CHW format and add batch dimension
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img


def benchmark_trt_engine(trt_inference, warmup_runs=10, benchmark_runs=100):
    """
    Benchmark TensorRT engine performance
    
    Args:
        trt_inference: TensorRTInference instance
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of benchmark iterations
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"{'='*70}")
    print(f"Benchmarking TensorRT Engine")
    print(f"{'='*70}")
    
    # Create dummy input
    dummy_input = np.random.randn(*trt_inference.input_shape).astype(np.float32)
    
    # Warmup
    print(f"Warming up ({warmup_runs} iterations)...")
    for _ in range(warmup_runs):
        _ = trt_inference.infer(dummy_input)
    
    # Benchmark
    print(f"Benchmarking ({benchmark_runs} iterations)...")
    times = []
    for i in range(benchmark_runs):
        start = time.time()
        _ = trt_inference.infer(dummy_input)
        times.append(time.time() - start)
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{benchmark_runs}")
    
    results = {
        "avg_time_ms": float(np.mean(times) * 1000),
        "std_time_ms": float(np.std(times) * 1000),
        "min_time_ms": float(np.min(times) * 1000),
        "max_time_ms": float(np.max(times) * 1000),
        "throughput_fps": float(1.0 / np.mean(times)),
        "benchmark_runs": benchmark_runs
    }
    
    print(f"\n✓ Benchmark Results:")
    print(f"  Average time: {results['avg_time_ms']:.2f} ms")
    print(f"  Min time: {results['min_time_ms']:.2f} ms")
    print(f"  Max time: {results['max_time_ms']:.2f} ms")
    print(f"  Throughput: {results['throughput_fps']:.2f} FPS\n")
    
    return results


def test_on_dataset(trt_inference, dataset_folder, class_names, output_dir="result"):
    """
    Test TensorRT model on images from dataset folder
    
    Args:
        trt_inference: TensorRTInference instance
        dataset_folder: Path to folder containing test images
        class_names: List of class names
        output_dir: Directory to save results
        
    Returns:
        Dictionary with test results
    """
    print(f"{'='*70}")
    print(f"Testing on Dataset")
    print(f"{'='*70}")
    
    dataset_path = Path(dataset_folder)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = [
        f for f in dataset_path.rglob('*')
        if f.suffix.lower() in image_extensions
    ]
    
    print(f"Found {len(image_paths)} images in {dataset_folder}\n")
    
    if len(image_paths) == 0:
        print("⚠ No images found in dataset folder")
        return {"images": [], "summary": {}}
    
    # Limit to first 50 images for speed
    max_images = min(50, len(image_paths))
    if len(image_paths) > max_images:
        print(f"Testing on first {max_images} images...\n")
        image_paths = image_paths[:max_images]
    
    results = {
        "images": [],
        "inference_times": [],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Process each image
    for idx, img_path in enumerate(image_paths, 1):
        try:
            # Preprocess
            input_data = preprocess_image(img_path)
            
            # Inference
            start = time.time()
            output = trt_inference.infer(input_data)
            inference_time = (time.time() - start) * 1000
            
            # Get predictions
            output_flat = output.flatten()
            probabilities = np.exp(output_flat) / np.sum(np.exp(output_flat))  # Softmax
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            # Store results
            image_result = {
                "image_name": img_path.name,
                "image_path": str(img_path),
                "predicted_class": int(predicted_class),
                "predicted_label": class_names[predicted_class],
                "confidence": float(confidence),
                "inference_time_ms": float(inference_time),
                "probabilities": {
                    class_names[i]: float(probabilities[i])
                    for i in range(len(class_names))
                }
            }
            
            results["images"].append(image_result)
            results["inference_times"].append(inference_time)
            
            # Print progress
            if idx % 10 == 0 or idx <= 5:
                print(f"[{idx}/{len(image_paths)}] {img_path.name}")
                print(f"  Prediction: {class_names[predicted_class]} ({confidence*100:.2f}%)")
                print(f"  Inference: {inference_time:.2f} ms")
                probs_str = " | ".join([f"{cn}: {probabilities[i]*100:.1f}%" 
                                       for i, cn in enumerate(class_names)])
                print(f"  Probabilities: {probs_str}\n")
                
        except Exception as e:
            print(f"✗ Error processing {img_path.name}: {e}\n")
            results["images"].append({
                "image_name": img_path.name,
                "error": str(e)
            })
    
    # Calculate summary statistics
    if results["inference_times"]:
        results["summary"] = {
            "total_images": len(image_paths),
            "successful": len(results["inference_times"]),
            "failed": len(image_paths) - len(results["inference_times"]),
            "avg_inference_time_ms": float(np.mean(results["inference_times"])),
            "min_inference_time_ms": float(np.min(results["inference_times"])),
            "max_inference_time_ms": float(np.max(results["inference_times"])),
            "class_distribution": {}
        }
        
        # Count predictions per class
        for class_name in class_names:
            count = sum(1 for img in results["images"] 
                       if img.get("predicted_label") == class_name)
            results["summary"]["class_distribution"][class_name] = count
        
        print(f"{'='*70}")
        print(f"Test Summary")
        print(f"{'='*70}")
        print(f"Total images: {results['summary']['total_images']}")
        print(f"Successful: {results['summary']['successful']}")
        print(f"Failed: {results['summary']['failed']}")
        print(f"Avg inference time: {results['summary']['avg_inference_time_ms']:.2f} ms")
        print(f"\nClass Distribution:")
        for class_name, count in results['summary']['class_distribution'].items():
            percentage = (count / results['summary']['successful'] * 100) if results['summary']['successful'] > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    return results


def save_results(conversion_info, benchmark_results, test_results, output_dir="result"):
    """Save all results to JSON and text files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine all results
    full_report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "conversion": conversion_info,
        "benchmark": benchmark_results,
        "dataset_test": test_results
    }
    
    # Save JSON
    json_path = output_path / f"trt_report_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(full_report, f, indent=4)
    
    # Save text report
    txt_path = output_path / f"trt_report_{timestamp}.txt"
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TENSORRT MODEL REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {full_report['timestamp']}\n\n")
        
        # Conversion info
        f.write("CONVERSION INFORMATION\n")
        f.write("-"*70 + "\n")
        for key, value in conversion_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Benchmark results
        f.write("BENCHMARK RESULTS\n")
        f.write("-"*70 + "\n")
        for key, value in benchmark_results.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Dataset test summary
        if "summary" in test_results and test_results["summary"]:
            f.write("DATASET TEST SUMMARY\n")
            f.write("-"*70 + "\n")
            for key, value in test_results["summary"].items():
                if key != "class_distribution":
                    f.write(f"{key}: {value}\n")
            
            f.write("\nClass Distribution:\n")
            for class_name, count in test_results["summary"]["class_distribution"].items():
                f.write(f"  {class_name}: {count}\n")
            f.write("\n")
        
        # Individual image results (first 20)
        if test_results.get("images"):
            f.write("SAMPLE IMAGE PREDICTIONS (First 20)\n")
            f.write("-"*70 + "\n")
            for img_result in test_results["images"][:20]:
                if "error" not in img_result:
                    f.write(f"\n{img_result['image_name']}:\n")
                    f.write(f"  Prediction: {img_result['predicted_label']} ({img_result['confidence']*100:.2f}%)\n")
                    f.write(f"  Inference time: {img_result['inference_time_ms']:.2f} ms\n")
    
    print(f"\n{'='*70}")
    print(f"Results Saved:")
    print(f"  JSON: {json_path}")
    print(f"  Text: {txt_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Configuration
    ONNX_PATH = "/home/dronex/Documents/TUBITAK-1002/model_simplified.onnx"  # Using simplified model
    TRT_PATH = "/home/dronex/Documents/TUBITAK-1002/model.trt"
    DATASET_FOLDER = "/home/dronex/Documents/TUBITAK-1002/dataset"
    OUTPUT_DIR = "result"
    CLASS_NAMES = ['fire', 'neither', 'smoke']
    
    # Step 1: Convert ONNX to TensorRT
    converter = TensorRTConverter(
        onnx_path=ONNX_PATH,
        trt_path=TRT_PATH,
        fp16_mode=True,
        max_workspace_size=1<<30  # 1GB
    )
    
    trt_engine_path = converter.build_engine()
    
    if trt_engine_path is None:
        print("✗ Failed to build TensorRT engine. Exiting.")
        exit(1)
    
    conversion_info = {
        "onnx_path": str(ONNX_PATH),
        "trt_path": str(trt_engine_path),
        "fp16_mode": converter.fp16_mode,
        "max_workspace_gb": converter.max_workspace_size / (1024**3)
    }
    
    # Step 2: Load TensorRT engine
    print(f"{'='*70}")
    print(f"Loading TensorRT Engine for Inference")
    print(f"{'='*70}\n")
    
    trt_inference = TensorRTInference(trt_engine_path)
    
    # Step 3: Benchmark performance
    benchmark_results = benchmark_trt_engine(
        trt_inference,
        warmup_runs=10,
        benchmark_runs=100
    )
    
    # Step 4: Test on dataset
    test_results = test_on_dataset(
        trt_inference,
        DATASET_FOLDER,
        CLASS_NAMES,
        OUTPUT_DIR
    )
    
    # Step 5: Save results
    save_results(conversion_info, benchmark_results, test_results, OUTPUT_DIR)
    
    print("\n✓ All tasks completed successfully!")
