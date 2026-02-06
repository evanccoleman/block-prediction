#!/usr/bin/env python3
"""
transfer_eval.py - Evaluate trained models on different dataset sizes

IMPORTANT NOTES ON TRANSFER LEARNING:
- diagonal_cnn CANNOT transfer: architecture has O(n) params specific to training size
- scalable_diagonal, conv_dense: CAN transfer in principle, but need .npz files
- image_resnet: CAN transfer and works with images only (recommended)

Usage:
    python transfer_eval.py --model image_resnet --model-path path/to/best_model.pt --test-dir path/to/test_data/

Run from block_jacobi_pytorch directory.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on different dataset')
    parser.add_argument('--model', type=str, required=True,
                        choices=['diagonal_cnn', 'scalable_diagonal', 'conv_dense', 'image_resnet'],
                        help='Model type')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to best_model.pt')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Path to test dataset directory')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--task', type=str, default='classification',
                        choices=['classification', 'regression'])
    args = parser.parse_args()
    
    # Import after path setup
    sys.path.insert(0, '.')
    from models import get_model
    from data import BlockJacobiDataset
    
    model_path = Path(args.model_path)
    test_dir = Path(args.test_dir)
    
    # Load config from training
    config_path = model_path.parent / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        train_matrix_size = config.get('matrix_size', 128)
        print(f"Loaded config: trained on {train_matrix_size}x{train_matrix_size} matrices")
    else:
        train_matrix_size = 128
        print(f"No config found, assuming train size = {train_matrix_size}")
    
    # Detect test matrix size
    test_jsons = list(test_dir.glob('*.json'))
    if test_jsons:
        with open(test_jsons[0]) as f:
            test_meta = json.load(f)
        test_matrix_size = test_meta.get('matrix_properties', {}).get('size', 'unknown')
        print(f"Test data: {test_matrix_size}x{test_matrix_size} matrices")
    else:
        test_matrix_size = 'unknown'
    
    # Check for transfer compatibility
    if args.model == 'diagonal_cnn' and train_matrix_size != test_matrix_size:
        print(f"\nWARNING: diagonal_cnn cannot transfer across sizes!")
        print(f"  Trained on n={train_matrix_size}, testing on n={test_matrix_size}")
        print(f"  The model architecture is size-specific (O(n) parameters).")
        print(f"  Use image_resnet for cross-size transfer learning.\n")
    
    # Create model with training size (architecture must match saved weights)
    print(f"\nCreating {args.model} model...")
    try:
        model = get_model(
            args.model,
            task=args.task,
            matrix_size=train_matrix_size
        )
    except Exception as e:
        print(f"Error creating model: {e}")
        sys.exit(1)
    
    # Load weights
    print(f"Loading weights from {model_path}...")
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)
    
    model.eval()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Load test data
    print(f"\nLoading test data from {test_dir}...")
    try:
        test_dataset = BlockJacobiDataset(
            data_dir=str(test_dir),
            task=args.task,
            model_type=args.model
        )
        print(f"Loaded {len(test_dataset)} test samples")
    except ValueError as e:
        print(f"Error: {e}")
        print("\nThis usually means the model needs .npz files that don't exist.")
        print("For large matrices (n>=2500), only image_resnet works.")
        sys.exit(1)
    
    if len(test_dataset) == 0:
        print("No test samples loaded!")
        sys.exit(1)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # Evaluate
    print(f"\nEvaluating...")
    if args.task == 'classification':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()
    
    correct = 0
    total = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                
                if args.task == 'classification':
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    all_preds.extend(predicted.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
                else:
                    total += labels.size(0)
                    
            except RuntimeError as e:
                if 'size mismatch' in str(e).lower() or 'shape' in str(e).lower():
                    print(f"\nSize mismatch error on batch {batch_idx}!")
                    print(f"This confirms the model cannot transfer to size {test_matrix_size}.")
                    print(f"Error: {e}")
                    sys.exit(1)
                raise
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {(batch_idx + 1) * args.batch_size} samples...")
    
    # Results
    print(f"\n{'='*50}")
    print(f"RESULTS: {args.model} trained on n={train_matrix_size}, tested on n={test_matrix_size}")
    print(f"{'='*50}")
    
    if total == 0:
        print("No samples evaluated!")
        sys.exit(1)
    
    avg_loss = total_loss / total
    
    if args.task == 'classification':
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"Test Loss: {avg_loss:.4f}")
        
        # Per-class accuracy
        from collections import Counter
        label_counts = Counter(all_labels)
        correct_per_class = Counter()
        for p, l in zip(all_preds, all_labels):
            if p == l:
                correct_per_class[l] += 1
        
        print(f"\nPer-class accuracy:")
        for cls in sorted(label_counts.keys()):
            cls_acc = correct_per_class[cls] / label_counts[cls] if label_counts[cls] > 0 else 0
            print(f"  Class {cls}: {cls_acc:.3f} ({correct_per_class[cls]}/{label_counts[cls]})")
    else:
        mae = avg_loss ** 0.5  # Approximate MAE from MSE
        print(f"Test Loss (MSE): {avg_loss:.4f}")
        print(f"Test MAE (approx): {mae:.4f}")


if __name__ == '__main__':
    main()
