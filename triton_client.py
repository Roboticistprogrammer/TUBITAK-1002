#!/usr/bin/env python3
"""
Triton Inference Client for Swin Transformer Model
Tests inference on images from the dataset folder
"""

import numpy as np
import cv2
import tritonclient.http as httpclient
from pathlib import Path
import json
from datetime import datetime


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess image for Swin Transformer model
    Expected input: [3, 224, 224] with values normalized
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Resize to 224x224
    img = cv2.resize(img, (224, 224))
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Normalize with ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    # Convert from HWC to CHW format
    img = np.transpose(img, (2, 0, 1))
    
    return img


def run_inference(client, image_path: str, model_name: str = "swin-transform"):
    """
    Run inference on a single image
    """
    # Preprocess image
    print(f"\nProcessing: {image_path}")
    preprocessed_image = preprocess_image(image_path)
    
    # Add batch dimension
    batch_input = np.expand_dims(preprocessed_image, axis=0)
    
    # Create input object for Triton
    inputs = []
    inputs.append(httpclient.InferInput("input", batch_input.shape, datatype="FP32"))
    inputs[0].set_data_from_numpy(batch_input, binary_data=True)
    
    # Define output
    outputs = []
    outputs.append(httpclient.InferRequestedOutput("output", binary_data=True))
    
    # Query the server
    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    
    # Get the output
    output = response.as_numpy("output")
    
    return output


def interpret_results(output: np.ndarray, class_names: list = None):
    """
    Interpret model output (assuming classification with 3 classes)
    """
    if class_names is None:
        class_names = ["Fire", "Smoke", "Neither"]  # Default class names
    
    # Get probabilities (apply softmax if needed)
    logits = output[0]
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()
    
    # Get prediction
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    
    result = {
        "predicted_class": class_names[predicted_class],
        "predicted_index": int(predicted_class),
        "confidence": float(confidence),
        "probabilities": {
            class_names[i]: float(probabilities[i]) 
            for i in range(len(class_names))
        }
    }
    
    return result


def main():
    # Configuration
    triton_url = "localhost:8000"
    model_name = "swin-transform"
    dataset_folder = Path("dataset")
    
    # Class names (adjust based on your model)
    class_names = ["Fire", "Smoke", "Neither"]
    
    # Connect to Triton server
    print(f"Connecting to Triton server at {triton_url}...")
    try:
        client = httpclient.InferenceServerClient(url=triton_url)
        
        # Check if server is live
        if not client.is_server_live():
            print("ERROR: Triton server is not live!")
            return
        
        # Check if server is ready
        if not client.is_server_ready():
            print("ERROR: Triton server is not ready!")
            return
        
        # Check if model is ready
        if not client.is_model_ready(model_name):
            print(f"ERROR: Model '{model_name}' is not ready!")
            return
        
        print(f"✓ Server is live and model '{model_name}' is ready\n")
        
        # Get model metadata
        metadata = client.get_model_metadata(model_name)
        print(f"Model metadata:")
        print(f"  Name: {metadata['name']}")
        print(f"  Versions: {metadata['versions']}")
        print(f"  Platform: {metadata['platform']}")
        
    except Exception as e:
        print(f"ERROR connecting to Triton server: {e}")
        print("\nMake sure the Triton server is running:")
        print("  docker run --runtime=nvidia --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \\")
        print("    -v ${PWD}/model_repository:/models \\")
        print("    nvcr.io/nvidia/deepstream:7.1-triton-multiarch \\")
        print("    tritonserver --model-repository=/models")
        return
    
    # Get list of test images
    image_files = list(dataset_folder.glob("*.jpg")) + list(dataset_folder.glob("*.png"))
    image_files = [f for f in image_files if f.is_file()]
    
    if not image_files:
        print(f"\nNo images found in {dataset_folder}/")
        return
    
    print(f"\nFound {len(image_files)} images to test\n")
    print("=" * 70)
    
    # Run inference on each image
    results = {}
    for image_path in image_files:
        try:
            # Run inference
            output = run_inference(client, str(image_path), model_name)
            
            # Interpret results
            result = interpret_results(output, class_names)
            
            # Store results
            results[image_path.name] = result
            
            # Print results
            print(f"Image: {image_path.name}")
            print(f"  Predicted: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"    {class_name}: {prob:.2%}")
            print("-" * 70)
            
        except Exception as e:
            print(f"ERROR processing {image_path.name}: {e}")
            print("-" * 70)
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"result/triton_inference_results_{timestamp}.json"
    
    # Create result directory if it doesn't exist
    Path("result").mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    print(f"✓ Processed {len(results)} images successfully")


if __name__ == "__main__":
    main()
