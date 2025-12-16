"""
Simple test script for inference on a single image
Usage: python test_inference.py --image path/to/fire.jpg --checkpoint path/to/model.pth
"""

import sys
import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import Config
from swin_transformer_pytorch.models.swin_transformer import SwinTransformer
from swin_transformer_pytorch.data_utils import get_transforms


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None
    
    model_config = checkpoint.get('config', {})
    
    # Create model with checkpoint config (or use default)
    model = SwinTransformer(
        hidden_dim=model_config.get('HIDDEN_DIM', Config.HIDDEN_DIM),
        layers=model_config.get('LAYERS', Config.LAYERS),
        heads=model_config.get('HEADS', Config.HEADS),
        channels=model_config.get('CHANNELS', Config.CHANNELS),
        num_classes=model_config.get('NUM_CLASSES', Config.NUM_CLASSES),
        head_dim=model_config.get('HEAD_DIM', Config.HEAD_DIM),
        window_size=model_config.get('WINDOW_SIZE', Config.WINDOW_SIZE),
        downscaling_factors=model_config.get('DOWNSCALING_FACTORS', Config.DOWNSCALING_FACTORS),
        relative_pos_embedding=model_config.get('RELATIVE_POS_EMBEDDING', Config.RELATIVE_POS_EMBEDDING)
    )
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None, None
    
    model = model.to(device)
    model.eval()
    
    return model, model_config


def infer_image(model, image_path, img_size, device, class_names):
    """Run inference on a single image"""
    
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"✓ Image loaded: {image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None
    
    # Transform
    transform = get_transforms('test', img_size, 'classification')
    image_tensor = transform(image).unsqueeze(0).to(device)
    print(f"✓ Image transformed: {image_tensor.shape}")
    
    # Inference
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
    
    pred_idx = logits.argmax(1).item()
    confidence = probs[0, pred_idx].item()
    
    result = {
        'class': class_names[pred_idx],
        'class_idx': pred_idx,
        'confidence': confidence,
        'logits': logits[0].cpu().numpy(),
        'probabilities': probs[0].cpu().numpy(),
    }
    
    return result, image


def print_results(result, class_names):
    """Print inference results"""
    print("\n" + "="*70)
    print("INFERENCE RESULTS")
    print("="*70)
    print(f"\n🎯 Predicted Class: {result['class']}")
    print(f"   Confidence: {result['confidence']*100:.2f}%")
    
    print(f"\n📊 Class Probabilities:")
    for i, (class_name, prob) in enumerate(zip(class_names, result['probabilities'])):
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"   {class_name:25s} {bar} {prob*100:6.2f}%")
    
    print("\n" + "="*70 + "\n")


def visualize_results(original_image, result, class_names, save_path=None):
    """Visualize inference results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Original image with prediction
    ax = axes[0]
    ax.imshow(original_image)
    ax.axis('off')
    ax.set_title(f'Input Image\nPredicted: {result["class"]} ({result["confidence"]*100:.1f}%)',
                 fontsize=12, fontweight='bold')
    
    # Right: Class probabilities
    ax = axes[1]
    colors = ['#FF6B6B', '#FFD93D', "#65C972"]
    probs = result['probabilities']
    bars = ax.bar(class_names, probs, color=colors[:len(class_names)], edgecolor='black', linewidth=2)
    
    # Highlight predicted class
    pred_idx = result['class_idx']
    bars[pred_idx].set_edgecolor('red')
    bars[pred_idx].set_linewidth(3)
    
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title('Class Probabilities', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    
    # Add percentage labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Test inference on a single image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python test_inference.py --image firee.jpg --checkpoint ../../checkpoints/best_model.pth
  
  # Show visualization
  python test_inference.py --image fire.jpg --checkpoint checkpoints/best_model.pth --visualize
  
  # Save results
  python test_inference.py --image fire.jpg --checkpoint checkpoints/best_model.pth --save results.png
        """)
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image (jpg, png, tif, etc.)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--visualize', action='store_true',
                       help='Display visualization')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save visualization (e.g., results.png)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--img-size', type=int, default=None,
                       help='Image size (will use checkpoint config or default 224)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"✓ Using device: {device}")
    
    # Check image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\n{'='*70}")
    print("INFERENCE TEST")
    print(f"{'='*70}\n")
    
    # Load model
    model, model_config = load_checkpoint(checkpoint_path, device)
    if model is None:
        print("❌ Failed to load model")
        return
    
    # Get image size
    img_size = args.img_size or model_config.get('IMG_SIZE', Config.IMG_SIZE)
    print(f"✓ Image size: {img_size}×{img_size}")
    
    # Run inference
    print(f"\n📸 Processing: {image_path.name}")
    result, original_image = infer_image(model, image_path, img_size, device, Config.CLASS_NAMES)
    
    if result is None:
        print("❌ Inference failed")
        return
    
    # Print results
    print_results(result, Config.CLASS_NAMES)
    
    # Visualize
    if args.visualize or args.save:
        visualize_results(original_image, result, Config.CLASS_NAMES, args.save)
    
    print("✓ Inference completed successfully!")


if __name__ == '__main__':
    main()
