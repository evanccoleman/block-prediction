#!/usr/bin/env python3
"""
Generate CNN Predictions for SuiteSparse Evaluation

Runs trained ImageResNet model on SuiteSparse matrix images and outputs 
predictions to a JSON file for use with evaluate_downstream.py.

Usage:
    python generate_predictions.py \
        --checkpoint ./experiments/full_comparison_n128/image_resnet_classification/image_resnet_classification_20260210_224203/best_model.pt \
        --images-dir ./suitesparse_eval/images \
        --output ./predictions_n128.json

    # Or run all three training sizes:
    for N in 128 500 1000; do
        CKPT=$(find ./experiments -path "*n${N}*image_resnet_classification*" -name "best_model.pt" | head -1)
        python generate_predictions.py --checkpoint "$CKPT" --images-dir ./suitesparse_eval/images --output ./predictions_n${N}.json
    done
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Import your model - adjust path if needed
try:
    from models import ImageResNet
except ImportError:
    print("ERROR: Could not import ImageResNet from models.py")
    print("Make sure you're running from the block_jacobi_pytorch directory")
    sys.exit(1)

FRACTION_CLASSES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CNN predictions for SuiteSparse matrices")
    
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--images-dir", "-i", type=str, required=True,
                        help="Directory containing matrix images (from suitesparse_eval/images)")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output JSON file for predictions")
    
    # Model configuration (should match training)
    parser.add_argument("--n-size", type=int, default=128,
                        help="Matrix size model was trained on (default: 128)")
    parser.add_argument("--num-classes", type=int, default=8,
                        help="Number of output classes (default: 8)")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Image size to resize to (default: 128)")
    
    # Processing
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["cuda", "cpu", "auto"],
                        help="Device for inference")
    
    return parser.parse_args()


def load_model(checkpoint_path: str, n_size: int, num_classes: int, device: torch.device):
    """Load trained ImageResNet model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model with same architecture as training
    model = ImageResNet(
        in_channels=1,
        n_size=n_size,
        num_classes=num_classes,
        task='classification'
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}, "
              f"val_loss={checkpoint.get('val_loss', '?')}")
    else:
        # Assume it's a raw state dict
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    return model


def predict_batch(model, image_paths: list, image_size: int, device: torch.device):
    """Run prediction on a batch of images."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    # Load and transform images
    batch = []
    valid_paths = []
    
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('L')  # Grayscale
            batch.append(transform(img))
            valid_paths.append(img_path)
        except Exception as e:
            print(f"  Warning: Could not load {img_path}: {e}")
    
    if not batch:
        return [], []
    
    batch_tensor = torch.stack(batch).to(device)
    
    with torch.no_grad():
        logits = model(batch_tensor)
        pred_classes = logits.argmax(dim=1).cpu().numpy()
        # Also get confidence scores
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        confidences = probs.max(axis=1)
    
    # Convert class indices to fractions
    predictions = [(FRACTION_CLASSES[c], float(conf)) 
                   for c, conf in zip(pred_classes, confidences)]
    
    return valid_paths, predictions


def main():
    args = parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Find all images
    images_dir = Path(args.images_dir)
    image_files = sorted(list(images_dir.glob("*.png")))
    
    if not image_files:
        print(f"No PNG files found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Load model
    model = load_model(args.checkpoint, args.n_size, args.num_classes, device)
    
    # Generate predictions
    predictions = {}
    confidence_scores = {}
    
    for i in range(0, len(image_files), args.batch_size):
        batch_files = image_files[i:i + args.batch_size]
        
        if i % 100 == 0:
            print(f"Processing {i}/{len(image_files)}...")
        
        valid_paths, batch_preds = predict_batch(
            model, batch_files, args.image_size, device
        )
        
        for img_path, (pred_frac, confidence) in zip(valid_paths, batch_preds):
            # Extract matrix_id from filename
            # e.g., "matrix_HB_bcsstk01.png" -> "HB_bcsstk01"
            matrix_id = img_path.stem
            if matrix_id.startswith("matrix_"):
                matrix_id = matrix_id[7:]  # Remove "matrix_" prefix
            
            predictions[matrix_id] = pred_frac
            confidence_scores[matrix_id] = confidence
    
    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "checkpoint": str(args.checkpoint),
        "n_size": args.n_size,
        "image_size": args.image_size,
        "num_predictions": len(predictions),
        "predictions": predictions,
        "confidence_scores": confidence_scores,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved {len(predictions)} predictions to {output_path}")
    
    # Print distribution summary
    from collections import Counter
    dist = Counter(predictions.values())
    print("\nPrediction distribution:")
    for frac in sorted(dist.keys()):
        print(f"  {frac:.2f}: {dist[frac]:3d} ({dist[frac]/len(predictions)*100:5.1f}%)")
    
    # Print confidence summary
    conf_values = list(confidence_scores.values())
    print(f"\nConfidence scores:")
    print(f"  Mean:   {np.mean(conf_values):.3f}")
    print(f"  Median: {np.median(conf_values):.3f}")
    print(f"  Min:    {np.min(conf_values):.3f}")
    print(f"  Max:    {np.max(conf_values):.3f}")


if __name__ == "__main__":
    main()
