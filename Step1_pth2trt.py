import torch
import torch.onnx
import torchvision.transforms as transforms
from pathlib import Path
import os
from PIL import Image
import sys

# Add Models directory to path
sys.path.insert(0, '/home/dronex/Documents/TUBITAK-1002/Models')

# Import your custom Swin Transformer model
from swin_transformer import SwinTransformer

def convert_pth_to_onnx(model_path, output_onnx_path, model_class, device='cuda'):
    """
    Convert PyTorch model to ONNX format with dynamic shapes support.
    
    Args:
        model_path: Path to .pth model file
        output_onnx_path: Output path for .onnx file
        model_class: Your model class (e.g., SwinTransformer)
        device: 'cuda' or 'cpu'
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Initialize model with config from checkpoint
    model = model_class(
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
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Define dummy input with dynamic batch size
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Export to ONNX with dynamic shapes
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        input_names=['image'],
        output_names=['output'],
        dynamic_axes={
            'image': {0: 'batch_size'},  
            'output': {0: 'batch_size'}
        },
        opset_version=12,
        do_constant_folding=True,
        verbose=True
    )
    
    print(f"✓ Model converted and saved to {output_onnx_path}")

def test_inference_on_dataset(model, dataset_folder, device='cuda', batch_size=4, class_names=None):
    """
    Test model inference on images in dataset folder.
    
    Args:
        model: PyTorch model
        dataset_folder: Path to folder containing images
        device: 'cuda' or 'cpu'
        batch_size: Batch size for inference
        class_names: List of class names for predictions
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(3)]
    
    model = model.to(device)
    model.eval()
    
    image_folder = Path(dataset_folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Gather all images
    image_paths = [
        f for f in image_folder.rglob('*') 
        if f.suffix.lower() in image_extensions
    ]
    
    print(f"\nFound {len(image_paths)} images in {dataset_folder}")
    print(f"Class names: {class_names}\n")
    
    # Inference loop
    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
                
                print(f"[{idx+1}/{len(image_paths)}] {img_path.name}")
                print(f"  Predicted: {class_names[predicted_class]} (confidence: {confidence:.2%})")
                print(f"  All probabilities: {', '.join([f'{class_names[i]}: {probabilities[0, i].item():.2%}' for i in range(len(class_names))])}")
                print()
                
            except Exception as e:
                print(f"✗ Error processing {img_path}: {e}")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "/home/dronex/Documents/TUBITAK-1002/Models/best_model_base.pth"
    ONNX_PATH = "/home/dronex/Documents/TUBITAK-1002/model.onnx"
    DATASET_FOLDER = "/home/dronex/Documents/TUBITAK-1002/dataset"
    DEVICE = 'cpu'  # Using CPU due to CUDA memory issues
    CLASS_NAMES = ['fire', 'neither', 'smoke']  # Based on your dataset
    
    print(f"Using device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Dataset folder: {DATASET_FOLDER}\n")
    
    # Step 1: Convert PTH to ONNX
    print("=" * 60)
    print("Step 1: Converting PyTorch model to ONNX format...")
    print("=" * 60)
    convert_pth_to_onnx(MODEL_PATH, ONNX_PATH, SwinTransformer, DEVICE)
    
    # Step 2: Load model and test inference
    print("\n" + "=" * 60)
    print("Step 2: Testing inference on dataset...")
    print("=" * 60)
    
    # Load checkpoint and initialize model
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
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
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best metric: {checkpoint.get('best_metric', 'unknown'):.2f}%\n")
    
    test_inference_on_dataset(model, DATASET_FOLDER, DEVICE, class_names=CLASS_NAMES)