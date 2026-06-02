
"""
Benchmark Script for SwinTransformer Model
Adapted from: https://github.com/NVIDIA/Torch-TensorRT/blob/master/notebooks/Resnet50-example.ipynb

Steps:
1. Load PyTorch SwinTransformer model
2. Benchmark on CPU
3. Benchmark on CUDA
4. Trace and compile with TensorRT
5. Benchmark TensorRT model
6. Make predictions with TensorRT model
"""

import torch
import torch.backends.cudnn as cudnn
import time
import numpy as np
import sys
import json
from datetime import datetime
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# Add Models directory to path
sys.path.insert(0, '/home/dronex/Documents/TUBITAK-1002/Models')
from Models.swin_transformer import SwinTransformer

cudnn.benchmark = True


def benchmark(model, device="cuda", input_shape=(32, 3, 224, 224), dtype='fp32', nwarmup=50, nruns=100):
    """
    Benchmark model inference speed.
    
    Args:
        model: PyTorch model to benchmark
        device: 'cuda' or 'cpu'
        input_shape: Input tensor shape (batch_size, channels, height, width)
        dtype: Data type ('fp32', 'fp16')
        nwarmup: Number of warmup iterations
        nruns: Number of timing iterations
    """
    input_data = torch.randn(input_shape)
    input_data = input_data.to(device)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            features = model(input_data)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            timings.append(end_time - start_time)
            
            if i % 10 == 0:
                print('Iteration %d/%d, ave batch time %.2f ms' % (i, nruns, np.mean(timings) * 1000))
    
    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print('Average batch time: %.2f ms' % (np.mean(timings) * 1000))
    
    return np.mean(timings) * 1000


def load_model(model_path, device='cuda'):
    """Load SwinTransformer model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    model = SwinTransformer(
        hidden_dim=config.get('HIDDEN_DIM', 128),
        layers=config.get('LAYERS', (2, 2, 18, 2)),
        heads=config.get('HEADS', (4, 8, 16, 32)),
        num_classes=config.get('NUM_CLASSES', 3),
        head_dim=config.get('HEAD_DIM', 32),
        window_size=config.get('WINDOW_SIZE', 7),
        downscaling_factors=config.get('DOWNSCALING_FACTORS', (4, 2, 2, 2)),
        relative_pos_embedding=config.get('RELATIVE_POS_EMBEDDING', True),
        channels=config.get('CHANNELS', 3)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best metric: {checkpoint.get('best_metric', 'unknown'):.2f}%")
    
    return model, config


def make_prediction(model, image_path, class_names, device='cuda'):
    """Make prediction on a single image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    probs, classes = torch.topk(probabilities, len(class_names))
    
    print(f"\nPredictions for {Path(image_path).name}:")
    for i in range(len(class_names)):
        probability = probs[i].item()
        class_label = class_names[int(classes[i])]
        print(f"  {probability*100:.2f}% {class_label}")


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "/home/dronex/Documents/TUBITAK-1002/Models/best_model_base.pth"
    BATCH_SIZE = 8  # Reduced for Jetson Orin compatibility
    INPUT_SHAPE = (BATCH_SIZE, 3, 224, 224)
    CLASS_NAMES = ['fire', 'neither', 'smoke']
    DATASET_FOLDER = "/home/dronex/Documents/TUBITAK-1002/dataset"
    OUTPUT_DIR = "result"
    
    # Try to find a test image
    dataset_path = Path(DATASET_FOLDER)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    test_images = [f for f in dataset_path.rglob('*') if f.suffix.lower() in image_extensions]
    TEST_IMAGE = str(test_images[0]) if test_images else None
    
    # Print system information
    print("=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Compute Capability: {torch.cuda.get_device_capability(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    
    print("=" * 70)
    print("STEP 1: Load SwinTransformer Model")
    print("=" * 70)
    model, config = load_model(MODEL_PATH, device='cpu')
    
    # Step 2: CPU Benchmarks
    print("\n" + "=" * 70)
    print("STEP 2: CPU Benchmarks")
    print("=" * 70)
    cpu_time = benchmark(model, device="cpu", input_shape=INPUT_SHAPE, nwarmup=10, nruns=50)
    
    # Step 3: Make predictions with CPU model on dataset
    print("\n" + "=" * 70)
    print("STEP 3: Testing CPU Model on Dataset Images")
    print("=" * 70)
    
    if TEST_IMAGE:
        # Test on multiple images from dataset
        test_count = min(10, len(test_images))
        print(f"Testing on {test_count} images from dataset...\n")
        
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(exist_ok=True)
        
        results = []
        for idx, img_path in enumerate(test_images[:test_count], 1):
            print(f"[{idx}/{test_count}] {img_path.name}")
            try:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to('cpu')
                
                with torch.no_grad():
                    start = time.time()
                    outputs = model(image_tensor)
                    inference_time = (time.time() - start) * 1000
                    
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                probs, classes = torch.topk(probabilities, len(CLASS_NAMES))
                
                result = {
                    "image": img_path.name,
                    "prediction": CLASS_NAMES[int(classes[0])],
                    "confidence": float(probs[0].item()),
                    "inference_time_ms": inference_time,
                    "probabilities": {
                        CLASS_NAMES[int(classes[i])]: float(probs[i].item())
                        for i in range(len(CLASS_NAMES))
                    }
                }
                results.append(result)
                
                print(f"  Prediction: {result['prediction']} ({result['confidence']*100:.2f}%)")
                print(f"  Inference: {inference_time:.2f} ms\n")
                
            except Exception as e:
                print(f"  Error: {e}\n")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_path / f"cpu_inference_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": "SwinTransformer",
                "device": "CPU",
                "results": results,
                "summary": {
                    "total_images": len(results),
                    "avg_inference_time_ms": np.mean([r["inference_time_ms"] for r in results]) if results else 0,
                    "avg_confidence": np.mean([r["confidence"] for r in results]) if results else 0
                }
            }, f, indent=4)
        
        print(f"✓ Results saved to {json_path}\n")
    else:
        print("⚠ No test images found in dataset folder")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("\n⚠ CUDA not available. Skipping GPU and TensorRT benchmarks.")
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"CPU Average Time: {cpu_time:.2f} ms")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Per-image time: {cpu_time/BATCH_SIZE:.2f} ms")
        exit(0)
    
    # Step 3: CUDA Benchmarks
    print("\n" + "=" * 70)
    print("STEP 4: CUDA Benchmarks")
    print("=" * 70)
    try:
        model = model.to("cuda")
        # Test with a single forward pass first
        print("Testing CUDA compatibility with single forward pass...")
        test_input = torch.randn(1, 3, 224, 224).to("cuda")
        with torch.no_grad():
            _ = model(test_input)
        print("✓ CUDA test successful, proceeding with full benchmark...")
        
        cuda_time = benchmark(model, device="cuda", input_shape=INPUT_SHAPE, nwarmup=50, nruns=100)
        cuda_available = True
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n⚠ CUDA Out of Memory Error!")
        print(f"Your Jetson Orin Nano may not have enough GPU memory for batch size {BATCH_SIZE}")
        print(f"Try reducing BATCH_SIZE in the configuration")
        cuda_available = False
        model = model.to("cpu")
    except RuntimeError as e:
        if "no kernel image is available" in str(e) or "cudaErrorNoKernelImageForDevice" in str(e):
            print(f"\n⚠ CUDA Kernel Compatibility Error!")
            print(f"Error: {e}")
            print("\n📋 DIAGNOSIS:")
            print("Your PyTorch installation doesn't have CUDA kernels for Jetson Orin Nano (Compute Capability 8.7)")
            print("\n💡 SOLUTIONS:")
            print("1. Install NVIDIA's JetPack-optimized PyTorch:")
            print("   - Uninstall current PyTorch: pip3 uninstall torch torchvision")
            print("   - Install from NVIDIA's wheel:")
            print("     wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl")
            print("     pip3 install torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl")
            print("\n2. Or use NVIDIA's PyTorch container:")
            print("   docker run -it --runtime nvidia nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3")
            print("\n3. Check NVIDIA's forum for latest wheels:")
            print("   https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048")
            print("\nContinuing with CPU-only benchmarks...")
        else:
            print(f"\n⚠ CUDA benchmark failed: {e}")
            print("Continuing with CPU-only benchmarks...")
        cuda_available = False
        model = model.to("cpu")
    except Exception as e:
        print(f"\n⚠ Unexpected CUDA error: {e}")
        print("Continuing with CPU-only benchmarks...")
        cuda_available = False
        model = model.to("cpu")
    
    if not cuda_available:
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"CPU Average Time: {cpu_time:.2f} ms")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Per-image time: {cpu_time/BATCH_SIZE:.2f} ms")
        exit(0)
    
    # Step 4: Trace and compile with TensorRT
    print("\n" + "=" * 70)
    print("STEP 5: Trace and Compile with TensorRT")
    print("=" * 70)
    
    try:
        import torch_tensorrt
        
        print("Tracing model with torch.jit.trace...")
        traced_model = torch.jit.trace(model, [torch.randn(INPUT_SHAPE).to("cuda")])
        
        print("Compiling model with torch_tensorrt...")
        trt_model = torch_tensorrt.compile(
            traced_model,
            inputs=[torch_tensorrt.Input(INPUT_SHAPE, dtype=torch.float32)],
            enabled_precisions={torch.float32}
        )
        print("✓ TensorRT compilation successful")
        
        # Step 5: TensorRT Benchmarks
        print("\n" + "=" * 70)
        print("STEP 6: TensorRT Benchmarks")
        print("=" * 70)
        trt_time = benchmark(trt_model, device="cuda", input_shape=INPUT_SHAPE, nwarmup=50, nruns=100)
        
        # Step 6: Make predictions with TensorRT model
        print("\n" + "=" * 70)
        print("STEP 7: Predictions with TensorRT Model")
        print("=" * 70)
        make_prediction(trt_model, TEST_IMAGE, CLASS_NAMES, device='cuda')
        
        # Summary
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"CPU Average Time:        {cpu_time:.2f} ms")
        print(f"CUDA Average Time:       {cuda_time:.2f} ms")
        print(f"TensorRT Average Time:   {trt_time:.2f} ms")
        print(f"\nSpeedup (CUDA vs CPU):   {cpu_time/cuda_time:.2f}x")
        print(f"Speedup (TRT vs CUDA):   {cuda_time/trt_time:.2f}x")
        print(f"Speedup (TRT vs CPU):    {cpu_time/trt_time:.2f}x")
        print(f"\nBatch size: {BATCH_SIZE}")
        print(f"Per-image time (TRT):    {trt_time/BATCH_SIZE:.2f} ms")
        
    except ImportError:
        print("\n⚠ torch_tensorrt not installed. Install with: pip install torch-tensorrt")
        print("Skipping TensorRT benchmarks.")
        
        # Make prediction with CUDA model instead
        print("\n" + "=" * 70)
        print("STEP 7: Predictions with CUDA Model")
        print("=" * 70)
        make_prediction(model, TEST_IMAGE, CLASS_NAMES, device='cuda')
        
        # Summary
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"CPU Average Time:  {cpu_time:.2f} ms")
        print(f"CUDA Average Time: {cuda_time:.2f} ms")
        print(f"\nSpeedup (CUDA vs CPU): {cpu_time/cuda_time:.2f}x")
        print(f"\nBatch size: {BATCH_SIZE}")
        print(f"Per-image time (CUDA): {cuda_time/BATCH_SIZE:.2f} ms")
    except Exception as e:
        print(f"\n⚠ TensorRT compilation failed: {e}")
        print("Using CUDA model for predictions...")
        
        print("\n" + "=" * 70)
        print("STEP 7: Predictions with CUDA Model")
        print("=" * 70)
        make_prediction(model, TEST_IMAGE, CLASS_NAMES, device='cuda')
        
        # Summary
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"CPU Average Time:  {cpu_time:.2f} ms")
        print(f"CUDA Average Time: {cuda_time:.2f} ms")
        print(f"\nSpeedup (CUDA vs CPU): {cpu_time/cuda_time:.2f}x")
        print(f"\nBatch size: {BATCH_SIZE}")
        print(f"Per-image time (CUDA): {cuda_time/BATCH_SIZE:.2f} ms")