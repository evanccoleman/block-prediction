#!/usr/bin/env python
"""
Prediction script for trained Block-Jacobi models.

Usage:
    python predict.py --model-path output/model/best_model.pt --input matrix_0.npz
    python predict.py --model-path output/model/best_model.pt --input-dir ./matrices/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from scipy import sparse

from models import DiagonalCNN, DenseNet, ImageCNN, ImageResNet
from data import (
    load_sparse_matrix,
    extract_diagonal_band,
    normalize_matrix,
    THRESHOLD_VALUES,
    THRESHOLD_TO_IDX,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run predictions with trained model')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config-path', type=str, default=None,
                        help='Path to config.json (auto-detected if not provided)')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to single input file (.npz or .png)')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Directory with input files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for predictions (JSON)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda, cpu, or auto)')
    
    return parser.parse_args()


def load_model(model_path: str, config_path: Optional[str] = None, device: torch.device = None):
    """Load a trained model from checkpoint."""
    # Find config
    model_dir = Path(model_path).parent
    if config_path is None:
        config_path = model_dir / 'config.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Determine number of classes
    if config['task'] == 'classification':
        num_classes = len(THRESHOLD_VALUES) if not config.get('multilabel', False) else config['n_size']
    else:
        num_classes = 1
    
    # Create model
    model_type = config['model']
    n_size = config['n_size']
    task = config['task']
    
    if model_type == 'diagonal_cnn':
        model = DiagonalCNN(
            n_size=n_size,
            band_width=config.get('band_width', 10),
            num_classes=num_classes,
            task=task
        )
    elif model_type == 'dense':
        model = DenseNet(n_size=n_size, num_classes=num_classes, task=task)
    elif model_type == 'image_cnn':
        model = ImageCNN(n_size=n_size, in_channels=1, num_classes=num_classes, task=task)
    elif model_type == 'image_resnet':
        model = ImageResNet(n_size=n_size, in_channels=1, num_classes=num_classes, task=task)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def preprocess_input(
    input_path: str,
    model_type: str,
    n_size: int = 128,
    band_width: int = 10,
) -> torch.Tensor:
    """Preprocess input file for the model."""
    input_path = Path(input_path)
    
    if input_path.suffix == '.npz':
        # Load sparse matrix
        matrix = load_sparse_matrix(input_path)
        dense = matrix.toarray().astype(np.float32)
        
        # Normalize to binary
        dense = normalize_matrix(dense, 'binary')
        
        if model_type == 'diagonal_cnn':
            processed = extract_diagonal_band(dense, band_width)
            return torch.tensor(processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            return torch.tensor(dense, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    elif input_path.suffix == '.png':
        from PIL import Image
        img = Image.open(input_path).convert('L')
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = 1.0 - img_array  # Invert
        return torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")


def predict_single(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    config: Dict,
    device: torch.device,
) -> Dict:
    """Make prediction for a single input."""
    model.eval()
    
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        
        if config['task'] == 'classification':
            if output.shape[1] == config['n_size']:
                # Multi-label (block starts)
                probs = torch.sigmoid(output).cpu().numpy()[0]
                block_starts = (probs > 0.5).astype(int)
                return {
                    'type': 'block_starts',
                    'block_starts': block_starts.tolist(),
                    'probabilities': probs.tolist(),
                    'num_blocks': int(block_starts.sum()),
                }
            else:
                # Single-label classification
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_idx = int(output.argmax(dim=1).item())
                pred_threshold = THRESHOLD_VALUES[pred_idx]
                return {
                    'type': 'threshold_classification',
                    'predicted_threshold': pred_threshold,
                    'predicted_index': pred_idx,
                    'probabilities': {str(t): float(p) for t, p in zip(THRESHOLD_VALUES, probs)},
                }
        else:
            # Regression
            pred_value = float(output.cpu().numpy()[0, 0])
            return {
                'type': 'threshold_regression',
                'predicted_value': pred_value,
            }


def main():
    args = parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model, config = load_model(args.model_path, args.config_path, device)
    print(f"Model type: {config['model']}, Task: {config['task']}")
    
    # Collect input files
    input_files = []
    if args.input:
        input_files.append(Path(args.input))
    if args.input_dir:
        input_dir = Path(args.input_dir)
        input_files.extend(sorted(input_dir.glob('*.npz')))
        input_files.extend(sorted(input_dir.glob('*.png')))
    
    if not input_files:
        print("No input files found!")
        return
    
    print(f"Processing {len(input_files)} files...")
    
    # Run predictions
    results = {}
    for input_path in input_files:
        print(f"  Processing: {input_path.name}")
        
        try:
            input_tensor = preprocess_input(
                str(input_path),
                config['model'],
                config['n_size'],
                config.get('band_width', 10),
            )
            
            prediction = predict_single(model, input_tensor, config, device)
            results[str(input_path)] = prediction
            
            # Print summary
            if prediction['type'] == 'block_starts':
                print(f"    -> {prediction['num_blocks']} blocks detected")
            elif prediction['type'] == 'threshold_classification':
                print(f"    -> Predicted threshold: {prediction['predicted_threshold']}")
            else:
                print(f"    -> Predicted value: {prediction['predicted_value']:.4f}")
                
        except Exception as e:
            print(f"    -> Error: {e}")
            results[str(input_path)] = {'error': str(e)}
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\nResults:")
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
