"""
Evaluation script for video action recognition
Computes multi-view accuracy and generates performance reports
"""
import os
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import VideoActionRecognitionModel


def evaluate_model(model, test_loader, device, num_classes=101):
    """
    Evaluate model on test set with detailed metrics.

    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to use
        num_classes: Number of classes

    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_video_ids = []

    # Store predictions for each view
    immediate_preds = []
    contextual_preds = []
    narrative_preds = []

    print(" Evaluating model...")

    with torch.no_grad():
        for clips, labels, metas in tqdm(test_loader, desc="Evaluating"):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            out = model(clips, return_attn=False)

            # Get predictions from each view
            immediate_logits = out["immediate_logits"]
            contextual_logits = out["contextual_logits"]
            narrative_logits = out["narrative_logits"]

            # Ensemble prediction (average logits)
            ensemble_logits = (immediate_logits + contextual_logits + narrative_logits) / 3.0
            ensemble_pred = torch.argmax(ensemble_logits, dim=1)

            # Individual view predictions
            immediate_pred = torch.argmax(immediate_logits, dim=1)
            contextual_pred = torch.argmax(contextual_logits, dim=1)
            narrative_pred = torch.argmax(narrative_logits, dim=1)

            # Store results
            all_preds.extend(ensemble_pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_video_ids.extend([meta.get('video_id', f'video_{i}') for i, meta in enumerate(metas)])

            immediate_preds.extend(immediate_pred.cpu().numpy())
            contextual_preds.extend(contextual_pred.cpu().numpy())
            narrative_preds.extend(narrative_pred.cpu().numpy())

    # Calculate metrics
    ensemble_acc = accuracy_score(all_labels, all_preds)
    immediate_acc = accuracy_score(all_labels, immediate_preds)
    contextual_acc = accuracy_score(all_labels, contextual_preds)
    narrative_acc = accuracy_score(all_labels, narrative_preds)

    print(f"\n Evaluation Results:")
    print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
    print(f"Immediate View Accuracy: {immediate_acc:.4f}")
    print(f"Contextual View Accuracy: {contextual_acc:.4f}")
    print(f"Narrative View Accuracy: {narrative_acc:.4f}")

    return {
        'ensemble_accuracy': ensemble_acc,
        'immediate_accuracy': immediate_acc,
        'contextual_accuracy': contextual_acc,
        'narrative_accuracy': narrative_acc,
        'predictions': all_preds,
        'labels': all_labels,
        'video_ids': all_video_ids
    }


def plot_confusion_matrix(labels, predictions, class_names, save_path=None):
    """Plot confusion matrix for evaluation results."""
    cm = confusion_matrix(labels, predictions)

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    plt.figure(figsize=(20, 16))
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Normalized)', fontsize=16)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('True Class', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def analyze_per_class_performance(labels, predictions, class_names):
    """Analyze per-class performance."""
    report = classification_report(labels, predictions, target_names=class_names,
                                 output_dict=True, zero_division=0)

    # Sort classes by F1-score
    class_f1_scores = [(class_name, metrics['f1-score'])
                       for class_name, metrics in report.items()
                       if class_name not in ['accuracy', 'macro avg', 'weighted avg']]
    class_f1_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n Top 10 Classes (by F1-score):")
    for i, (class_name, f1) in enumerate(class_f1_scores[:10]):
        print(f"{i+1:2d}. {class_name:20s}: {f1:.4f}")

    print(f"\n Bottom 10 Classes (by F1-score):")
    for i, (class_name, f1) in enumerate(class_f1_scores[-10:]):
        print(f"{i+1:2d}. {class_name:20s}: {f1:.4f}")

    return report


def save_evaluation_results(results, report, save_dir):
    """Save evaluation results to files."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save numerical results
    with open(save_dir / "evaluation_results.json", 'w') as f:
        json.dump({
            'ensemble_accuracy': float(results['ensemble_accuracy']),
            'immediate_accuracy': float(results['immediate_accuracy']),
            'contextual_accuracy': float(results['contextual_accuracy']),
            'narrative_accuracy': float(results['narrative_accuracy']),
            'classification_report': report
        }, f, indent=2)

    # Save predictions
    np.save(save_dir / "predictions.npy", results['predictions'])
    np.save(save_dir / "labels.npy", results['labels'])

    with open(save_dir / "video_ids.txt", 'w') as f:
        for vid_id in results['video_ids']:
            f.write(f"{vid_id}\n")

    print(f" Results saved to {save_dir}")


def load_trained_model(model_path, device, num_classes=101, feature_dim=2048, spatial_size=4,
                      embed_dim=384, spatial_layers=4, spatial_heads=6,
                      temporal_layers=3, temporal_heads=6, max_T=16):
    """Load trained model from checkpoint."""
    model = VideoActionRecognitionModel(
        feature_dim=feature_dim,
        spatial_size=spatial_size,
        embed_dim=embed_dim,
        spatial_layers=spatial_layers,
        spatial_heads=spatial_heads,
        temporal_layers=temporal_layers,
        temporal_heads=temporal_heads,
        mlp_ratio=4.0,
        dropout=0.1,
        max_T=max_T,
        num_classes=num_classes
    ).to(device)

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model with {num_params:,} parameters from {model_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate video action recognition model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--num_classes', type=int, default=101, help='Number of classes')
    parser.add_argument('--T', type=int, default=64, help='Number of temporal frames')
    parser.add_argument('--out_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    print(" Loading trained model...")
    if not os.path.exists(args.model_path):
        print(f" Model not found: {args.model_path}")
        return

    model = load_trained_model(args.model_path, device, num_classes=args.num_classes, max_T=args.T)

    # TODO: Replace with your own DataLoader
    # test_loader = create_test_dataloader(args.data_dir, args.batch_size, args.T)
    print("Please implement your own test DataLoader in this script")
    print("See README.md for dataset format requirements")
    return

    # Generate class names (replace with your actual class names)
    class_names = [f"Class_{i}" for i in range(args.num_classes)]

    # Evaluate model
    results = evaluate_model(model, test_loader, device, num_classes=args.num_classes)

    # Analyze results
    print("\n Analyzing per-class performance...")
    report = analyze_per_class_performance(results['labels'], results['predictions'], class_names)

    # Save results
    save_evaluation_results(results, report, args.out_dir)

    # Plot confusion matrix
    plot_confusion_matrix(results['labels'], results['predictions'], class_names,
                         save_path=f"{args.out_dir}/confusion_matrix.png")

    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    accuracies = [
        results['immediate_accuracy'],
        results['contextual_accuracy'],
        results['narrative_accuracy'],
        results['ensemble_accuracy']
    ]
    views = ['Immediate', 'Contextual', 'Narrative', 'Ensemble']

    bars = plt.bar(views, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    plt.title('Multi-View Performance', fontsize=16)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1.0)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n Evaluation completed!")
    print(f" Final Ensemble Accuracy: {results['ensemble_accuracy']:.4f}")
    print(f"All results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
