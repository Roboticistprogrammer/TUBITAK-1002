import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, 'Models')
from swin_transformer import SwinTransformer

def test_pytorch_model(model_path, input_size=(1, 3, 224, 224)):
    """Test PyTorch custom Swin Transformer model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"Using {device} device")
    
    try:
        # Load checkpoint
        print("Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract config and model state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("✓ Found checkpoint with metadata")
            state_dict = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
            print(f"  Config: {config}")
        else:
            print("✓ Using checkpoint as direct state dict")
            state_dict = checkpoint
            config = {}
        
        # Create model with config parameters or defaults
        print("\nCreating custom SwinTransformer model...")
        model = SwinTransformer(
            hidden_dim=config.get('HIDDEN_DIM', config.get('hidden_dim', 128)),
            layers=tuple(config.get('LAYERS', config.get('layers', (2, 2, 18, 2)))),
            heads=tuple(config.get('HEADS', config.get('heads', (4, 8, 16, 32)))),
            num_classes=config.get('NUM_CLASSES', config.get('num_classes', 2)),
            window_size=config.get('WINDOW_SIZE', config.get('window_size', 7)),
            relative_pos_embedding=config.get('RELATIVE_POS_EMBEDDING', config.get('relative_pos_embedding', True))
        )
        
        # Load state dict
        model.load_state_dict(state_dict)
        
        # Try to move to GPU
        try:
            model = model.to(device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("WARNING: GPU out of memory. Falling back to CPU.")
                device = "cpu"
                model = model.to(device)
            else:
                raise
        
        model.eval()
        
        print(f"✓ Model loaded successfully from {model_path}")
        print(f"✓ Model device: {next(model.parameters()).device}")
        print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test inference
        print(f"\nRunning inference with input shape: {input_size}")
        with torch.no_grad():
            dummy_input = torch.randn(input_size).to(device)
            
            try:
                output = model(dummy_input)
                print(f"✓ Input shape: {dummy_input.shape}")
                print(f"✓ Output shape: {output.shape}")
                
                # Get predictions
                if output.shape[1] > 1:
                    top_values, top_indices = torch.topk(output, min(5, output.shape[1]))
                    print(f"✓ Top-5 predictions:")
                    for i, (idx, val) in enumerate(zip(top_indices[0], top_values[0]), 1):
                        print(f"  {i}. Class {idx.item()}: {val.item():.4f}")
                else:
                    print(f"✓ Output value: {output[0, 0].item():.4f}")
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"WARNING: GPU out of memory during inference.")
                    print(f"Error: {e}")
                    return None
                else:
                    raise
        
        return model
        
    except Exception as e:
        print(f"✗ ERROR: Failed to test model - {e}")
        import traceback
        traceback.print_exc()
        return None

def Onnx2Trt(onnx_path,trt_path):
    # Load the ONNX model
    onnx_model_path = f"{onnx_path}.onnx"
    trt_model_path = f"{trt_path}.trt"

    # Create a TensorRT builder and network
    trt_logger = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(trt_logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network,
                                                                                                               trt_logger) as parser:
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 28
        builder.max_batch_size = 1

        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            trt_model_path = f"{trt_path}_FP16.trt"
            print("In FP16!!!")
        else:
            print("Warning: Fast FP16 not supported on this platform. Using default precision.")


        with open(onnx_model_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
        print('Completed parsing of ONNX file')
        plan = builder.build_serialized_network(network, config)
        with trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        with open(trt_model_path, "wb") as f:
            f.write(engine.serialize())

if __name__=='__main__':
    # Test PyTorch model
    print("="*50)
    print("Testing PyTorch Swin Transformer Model")
    print("="*50)
    test_pytorch_model('Models/best_model_base.pth')







