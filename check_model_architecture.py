#!/usr/bin/env python3
"""
Diagnostic script to analyze your model checkpoint architecture
"""

import torch
import json

def analyze_checkpoint(checkpoint_path):
    """Analyze and display checkpoint structure"""
    print("=" * 60)
    print(f"Analyzing checkpoint: {checkpoint_path}")
    print("=" * 60)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return
    
    # Check checkpoint structure
    print("\nCheckpoint top-level keys:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  {key}: dict with {len(checkpoint[key])} items")
        elif isinstance(checkpoint[key], torch.Tensor):
            print(f"  {key}: tensor {checkpoint[key].shape}")
        else:
            print(f"  {key}: {type(checkpoint[key])}")
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        print("\n✓ Found 'model_state_dict' in checkpoint")
        state_dict = checkpoint['model_state_dict']
    else:
        print("\n✓ Using checkpoint as state_dict directly")
        state_dict = checkpoint
    
    # Analyze model architecture
    print(f"\nModel state_dict has {len(state_dict)} parameters:")
    
    # Group by stage/module
    modules = {}
    for key in state_dict.keys():
        module_name = key.split('.')[0]
        if module_name not in modules:
            modules[module_name] = []
        modules[module_name].append(key)
    
    print("\nModule breakdown:")
    for module_name in sorted(modules.keys()):
        print(f"  {module_name}: {len(modules[module_name])} parameters")
    
    # Show sample keys
    print("\nSample parameter names (first 15):")
    for i, key in enumerate(list(state_dict.keys())[:15]):
        shape = state_dict[key].shape
        print(f"  {i+1}. {key}: {shape}")
    
    # Detect model type
    print("\n" + "=" * 60)
    print("MODEL TYPE DETECTION:")
    print("=" * 60)
    
    keys_str = ' '.join(state_dict.keys())
    
    if 'stage' in keys_str:
        print("\n✓ Custom Swin Transformer detected (uses stage1, stage2, etc.)")
        print("  This is NOT the torchvision swin_b model!")
        print("  You need to find/define your custom model class")
    elif 'features.' in keys_str:
        print("\n✓ Standard torchvision Swin Transformer detected")
        print("  This is compatible with swin_b() from torchvision")
    elif 'encoder' in keys_str:
        print("\n✓ Vision Transformer or custom encoder detected")
    else:
        print("\n? Unknown model architecture")
    
    # Save analysis to file
    analysis = {
        'num_parameters': len(state_dict),
        'modules': {m: len(params) for m, params in modules.items()},
        'sample_keys': list(state_dict.keys())[:20]
    }
    
    with open('model_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print("\n✓ Analysis saved to model_analysis.json")

if __name__ == '__main__':
    analyze_checkpoint('Models/best_model.pth')
