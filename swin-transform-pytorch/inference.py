"""
Inference script for single image or batch prediction
"""

import sys
import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from swin_transformer_pytorch.models.swin_transformer import SwinTransformer
from swin_transformer_pytorch.data_utils import get_transforms


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['config']
    
    model = SwinTransformer(
        hidden_dim=model_config['HIDDEN_DIM'],
        layers=model_config['LAYERS'],
        heads=model_config['HEADS'],
        channels=model_config['CHANNELS'],
        num_classes=model_config['NUM_CLASSES'],
        head_dim=model_config['HEAD_DIM'],
        window_size=model_config['WINDOW_SIZE'],
        downscaling_factors=model_config['DOWNSCALING_FACTORS'],
        relative_pos_embedding=model_config['RELATIVE_POS_EMBEDDING']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, model_config


def predict_image(model, image_path, transform, device, class_names):
    """Predict single image"""
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred_idx = output.argmax(1).item()
        confidence = probabilities[0, pred_idx].item()
    
    prediction = {
        'class': class_names[pred_idx],
        'class_idx': pred_idx,
        'confidence': confidence,
        'probabilities': {
            class_names[i]: probabilities[0, i].item() 
            for i in range(len(class_names))
        }
    }
    
    return prediction, image


def visualize_prediction(image, prediction, save_path=None):
    """Visualize prediction result"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Image')
    
    # Show prediction probabilities
    class_names = list(prediction['probabilities'].keys())
    probs = list(prediction['probabilities'].values())
    colors = ['#FF6B6B', '#FFD93D', '#6BCB77']
    
    bars = ax2.barh(class_names, probs, color=colors)
    ax2.set_xlabel('Probability')
    ax2.set_xlim([0, 1])
    ax2.set_title(f'Prediction: {prediction["class"]} ({prediction["confidence"]*100:.1f}%)')
    
    # Highlight predicted class
    for i, bar in enumerate(bars):
        if class_names[i] == prediction['class']:
            bar.set_edgecolor('black')
            bar.set_linewidth(3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Inference with trained Swin Transformer')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize prediction')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found at {args.image}")
        return
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, model_config = load_model(args.checkpoint, device)
    print("Model loaded successfully!")
    
    # Get transform
    transform = get_transforms('test', model_config['IMG_SIZE'], 'classification')
    
    # Class names
    class_names = Config.CLASS_NAMES
    
    # Predict
    print(f"\nPredicting: {args.image}")
    prediction, original_image = predict_image(
        model, args.image, transform, device, class_names
    )
    
    # Print results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"Predicted Class: {prediction['class']}")
    print(f"Confidence: {prediction['confidence']*100:.2f}%")
    print("\nClass Probabilities:")
    for class_name, prob in prediction['probabilities'].items():
        print(f"  {class_name:25s}: {prob*100:6.2f}%")
    print("="*70)
    
    # Visualize
    if args.visualize:
        visualize_prediction(original_image, prediction, args.save)


if __name__ == '__main__':
    main()
