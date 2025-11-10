"""
Evaluation and Inference script for trained Swin Transformer model
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from swin_transformer_pytorch.models.swin_transformer import SwinTransformer
from swin_transformer_pytorch.data_utils import create_dataloaders, FASSDDataset, get_transforms


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['config']
    
    # Create model
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
    
    print(f"Model loaded successfully (Epoch: {checkpoint['epoch']})")
    return model, model_config


@torch.no_grad()
def evaluate_model(model, dataloader, device, config):
    """Evaluate model and return predictions"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("\nEvaluating model...")
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    return all_predictions, all_labels, all_probabilities


def compute_metrics(predictions, labels, class_names):
    """Compute detailed metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    
    # Per-class metrics
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    
    # Get unique labels present in the data
    unique_labels = np.unique(np.concatenate([labels, predictions]))
    target_names_subset = [class_names[i] for i in unique_labels]
    
    report = classification_report(labels, predictions, 
                                   labels=unique_labels,
                                   target_names=target_names_subset, 
                                   digits=4, 
                                   zero_division=0)
    print(report)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report
    }


def plot_confusion_matrix(predictions, labels, class_names, save_path=None):
    """Plot confusion matrix"""
    # Get unique labels present in the data
    unique_labels = np.unique(np.concatenate([labels, predictions]))
    target_names_subset = [class_names[i] for i in unique_labels]
    
    cm = confusion_matrix(labels, predictions, labels=unique_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names_subset, yticklabels=target_names_subset)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved: {save_path}")
    
    plt.show()


def plot_per_class_accuracy(predictions, labels, class_names, save_path=None):
    """Plot per-class accuracy"""
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    for label, pred in zip(labels, predictions):
        class_total[label] += 1
        if label == pred:
            class_correct[label] += 1
    
    class_accuracies = [100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                       for i in range(len(class_names))]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_accuracies, color=['#FF6B6B', '#FFD93D', '#6BCB77'])
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    plt.ylim([0, 105])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class accuracy plot saved: {save_path}")
    
    plt.show()


def plot_probability_distribution(probabilities, predictions, labels, class_names, save_path=None):
    """Plot probability distributions for correct and incorrect predictions"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Correct predictions
    correct_mask = predictions == labels
    correct_probs = probabilities[correct_mask]
    correct_max_probs = correct_probs.max(axis=1)
    
    axes[0].hist(correct_max_probs, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Prediction Probability')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Correct Predictions (n={len(correct_max_probs)})')
    axes[0].axvline(correct_max_probs.mean(), color='red', linestyle='--', 
                    label=f'Mean: {correct_max_probs.mean():.3f}')
    axes[0].legend()
    
    # Incorrect predictions
    incorrect_mask = predictions != labels
    incorrect_probs = probabilities[incorrect_mask]
    incorrect_max_probs = incorrect_probs.max(axis=1)
    
    axes[1].hist(incorrect_max_probs, bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Prediction Probability')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Incorrect Predictions (n={len(incorrect_max_probs)})')
    if len(incorrect_max_probs) > 0:
        axes[1].axvline(incorrect_max_probs.mean(), color='blue', linestyle='--', 
                       label=f'Mean: {incorrect_max_probs.mean():.3f}')
        axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probability distribution plot saved: {save_path}")
    
    plt.show()


def visualize_predictions(model, dataset, device, class_names, num_samples=9, save_path=None):
    """Visualize model predictions on sample images"""
    import random
    from PIL import Image
    
    model.eval()
    
    # Randomly select samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    for idx, sample_idx in enumerate(indices):
        # Get sample
        sample = dataset.samples[sample_idx]
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Get prediction
        transform = get_transforms('test', dataset.transform.transforms[0].size[0], 'classification')
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)
            pred_idx = output.argmax(1).item()
            confidence = prob[0, pred_idx].item()
        
        true_label = dataset.CLASS_TO_IDX.get(sample['image_label'], 2)
        pred_label = class_names[pred_idx]
        true_label_name = class_names[true_label]
        
        # Plot
        axes[idx].imshow(image)
        axes[idx].axis('off')
        
        color = 'green' if pred_idx == true_label else 'red'
        title = f"True: {true_label_name}\nPred: {pred_label} ({confidence*100:.1f}%)"
        axes[idx].set_title(title, color=color, fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained Swin Transformer model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize sample predictions')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to results directory')
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        checkpoint_path = config.CHECKPOINT_DIR / args.checkpoint
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    model, model_config = load_model(checkpoint_path, device)
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    _, val_loader, test_loader = create_dataloaders(
        root_dir=config.DATASET_ROOT,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
        img_size=model_config['IMG_SIZE'],
        annotation_format=config.ANNOTATION_FORMAT,
        task=config.TASK,
        pin_memory=False
    )
    
    if args.split == 'val':
        dataloader = val_loader
    else:
        dataloader = test_loader
    
    # Evaluate
    predictions, labels, probabilities = evaluate_model(model, dataloader, device, config)
    
    # Compute metrics
    metrics = compute_metrics(predictions, labels, config.CLASS_NAMES)
    
    # Create results directory
    results_dir = config.RESULTS_DIR / f'evaluation_{args.split}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    save_path = results_dir / 'metrics.txt' if args.save_plots else None
    if save_path:
        with open(save_path, 'w') as f:
            f.write(f"Evaluation on {args.split} set\n")
            f.write("="*70 + "\n")
            f.write(f"Accuracy:  {metrics['accuracy']*100:.2f}%\n")
            f.write(f"Precision: {metrics['precision']*100:.2f}%\n")
            f.write(f"Recall:    {metrics['recall']*100:.2f}%\n")
            f.write(f"F1 Score:  {metrics['f1']*100:.2f}%\n\n")
            f.write(metrics['report'])
        print(f"\nMetrics saved to: {save_path}")
    
    # Plot confusion matrix
    cm_path = results_dir / 'confusion_matrix.png' if args.save_plots else None
    plot_confusion_matrix(predictions, labels, config.CLASS_NAMES, cm_path)
    
    # Plot per-class accuracy
    acc_path = results_dir / 'per_class_accuracy.png' if args.save_plots else None
    plot_per_class_accuracy(predictions, labels, config.CLASS_NAMES, acc_path)
    
    # Plot probability distribution
    prob_path = results_dir / 'probability_distribution.png' if args.save_plots else None
    plot_probability_distribution(probabilities, predictions, labels, config.CLASS_NAMES, prob_path)
    
    # Visualize predictions
    if args.visualize:
        dataset = FASSDDataset(
            root_dir=config.DATASET_ROOT,
            split=args.split,
            annotation_format=config.ANNOTATION_FORMAT,
            task=config.TASK
        )
        vis_path = results_dir / 'sample_predictions.png' if args.save_plots else None
        visualize_predictions(model, dataset, device, config.CLASS_NAMES, num_samples=9, save_path=vis_path)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETED!")
    print("="*70)


if __name__ == '__main__':
    main()
