#!/usr/bin/env python3
"""
Re-export ONNX model for Triton Inference Server.
This script creates a complete ONNX model without external data file dependencies.
"""

import torch
import torch.onnx
import onnx
import sys
import os
from pathlib import Path

# Add Models directory to path
sys.path.insert(0, '/home/dronex/Documents/TUBITAK-1002/Models')

from Models.swin_transformer import SwinTransformer


def remove_scatternd_reduction_attr(onnx_path: str) -> int:
    """
    TensorRT 10.8 fails when ScatterND contains the optional `reduction` attribute,
    even when it is set to "none". Remove this attribute for compatibility.
    """
    model = onnx.load(onnx_path)
    modified = 0

    for node in model.graph.node:
        if node.op_type != "ScatterND":
            continue

        has_reduction = any(attr.name == "reduction" for attr in node.attribute)
        if has_reduction:
            kept_attrs = [attr for attr in node.attribute if attr.name != "reduction"]
            del node.attribute[:]
            node.attribute.extend(kept_attrs)
            modified += 1

    if modified > 0:
        onnx.save(model, onnx_path)

    return modified

def export_onnx_for_triton():
    """
    Export PyTorch model to ONNX with all weights embedded (no external data).
    """
    model_path = "Models/best_model.pth"
    output_dir = "model_repository/swin-transform/1"
    output_onnx = os.path.join(output_dir, "model.onnx")
    
    print("=" * 70)
    print("Re-exporting ONNX Model for Triton Inference Server")
    print("=" * 70)
    
    # Load checkpoint on CPU to avoid CUDA issues
    print(f"\n1. Loading PyTorch model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint.get('config', {})
    
    print(f"   Model config: {config}")
    
    # Initialize model
    print("\n2. Initializing Swin Transformer model...")
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
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    print("\n3. Creating dummy input (1, 3, 224, 224)...")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Remove old files if they exist
    if os.path.exists(output_onnx):
        print(f"\n4. Removing old ONNX file: {output_onnx}")
        os.remove(output_onnx)
    
    # Also remove .data file if it exists
    data_file = output_onnx + ".data"
    if os.path.exists(data_file):
        print(f"   Removing old data file: {data_file}")
        os.remove(data_file)
    
    print(f"\n5. Exporting to ONNX (this may take a minute)...")
    print(f"   Output: {output_onnx}")
    
    # Export with weights embedded (NOT external)
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=16,  # Use opset 16 for better compatibility
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    # Check file size
    file_size_mb = os.path.getsize(output_onnx) / (1024 * 1024)
    print(f"\n✓ ONNX export complete!")
    print(f"  File: {output_onnx}")
    print(f"  Size: {file_size_mb:.1f} MB")
    
    if file_size_mb < 100:
        print("\n⚠ WARNING: File size is suspiciously small!")
        print("  Expected ~330MB for this model.")
        print("  Checking for external data file...")
        
        if os.path.exists(data_file):
            data_size_mb = os.path.getsize(data_file) / (1024 * 1024)
            print(f"  ✓ Found {data_file} ({data_size_mb:.1f} MB)")
            print(f"  Total size: {file_size_mb + data_size_mb:.1f} MB")
        else:
            print(f"  ✗ No external data file found!")
            return False
    
    print("\n6. Patching ONNX graph for TensorRT compatibility...")
    patched_nodes = remove_scatternd_reduction_attr(output_onnx)
    if patched_nodes > 0:
        print(f"  ✓ Removed unsupported ScatterND reduction attribute from {patched_nodes} node(s)")
    else:
        print("  No ScatterND reduction attributes found")

    # Verify the model can be loaded
    print("\n7. Verifying ONNX model...")
    try:
        onnx_model = onnx.load(output_onnx)
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX model is valid!")
        
        # Print input/output info for config.pbtxt
        print("\n8. Model I/O Information (for config.pbtxt):")
        print(f"  Input name: {onnx_model.graph.input[0].name}")
        input_shape = [dim.dim_value if dim.dim_value > 0 else -1 
                      for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        print(f"  Input shape: {input_shape}")
        print(f"  Input type: {onnx_model.graph.input[0].type.tensor_type.elem_type}")
        
        print(f"\n  Output name: {onnx_model.graph.output[0].name}")
        output_shape = [dim.dim_value if dim.dim_value > 0 else -1 
                       for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim]
        print(f"  Output shape: {output_shape}")
        print(f"  Output type: {onnx_model.graph.output[0].type.tensor_type.elem_type}")
        
    except Exception as e:
        print(f"  ✗ ONNX verification failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("SUCCESS! ONNX model is ready for Triton")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Create config.pbtxt file (I'll create it for you)")
    print("2. Retry the docker run command")
    
    return True

def create_triton_config():
    """
    Create config.pbtxt for Triton Inference Server.
    """
    config_path = "model_repository/swin-transform/config.pbtxt"
    
    config_content = """name: "swin-transform"
platform: "onnxruntime_onnx"
max_batch_size: 4

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 3 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"\n✓ Created Triton config file: {config_path}")
    print("\nConfig contents:")
    print("-" * 50)
    print(config_content)
    print("-" * 50)

if __name__ == "__main__":
    success = export_onnx_for_triton()
    
    if success:
        create_triton_config()
        print("\n✓ All files ready for Triton Inference Server!")
    else:
        print("\n✗ ONNX export failed. Please check the errors above.")
        sys.exit(1)
