#!/usr/bin/env python
"""
Training script for Block-Jacobi preconditioner prediction models.

Supports three model architectures:
1. DiagonalCNN: CNN on diagonal band (Götz & Anzt approach)
2. DenseNet: Dense network on full matrix
3. ImageCNN/ImageResNet: CNN on matrix image

Usage:
    python train.py --model diagonal_cnn --task classification --epochs 50
    python train.py --model dense --task regression --epochs 100
    python train.py --model image_resnet --task classification --epochs 50
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from models import get_model, DiagonalCNN, DenseNet, ImageCNN, ImageResNet
from data import (
    BlockJacobiDataset, 
    SyntheticBlockJacobiDataset,
    create_dataloaders,
    THRESHOLD_VALUES,
    denormalize_regression_prediction,
    REGRESSION_RANGE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Block-Jacobi preconditioner prediction models'
    )
    
    # Model configuration
    parser.add_argument('--model', type=str, default='diagonal_cnn',
                        choices=['diagonal_cnn', 'dense', 'image_cnn', 'image_resnet'],
                        help='Model architecture')
    parser.add_argument('--task', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help='Task type')
    
    # Data configuration
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory with matrix data (if None, use synthetic)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data generation')
    parser.add_argument('--num-samples', type=int, default=3000,
                        help='Number of synthetic samples')
    parser.add_argument('--n-size', type=int, default=128,
                        help='Matrix size')
    parser.add_argument('--band-width', type=int, default=10,
                        help='Diagonal band half-width')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.02,
                        help='L2 regularization weight')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split fraction')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Output directory for models and logs')
    parser.add_argument('--save-best', action='store_true', default=True,
                        help='Save best model based on validation loss')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda, cpu, or auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Regularization (NEW)
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing factor (0 = disabled)')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate (overrides model default)')
    parser.add_argument('--early-stopping', type=int, default=0,
                        help='Early stopping patience (0 = disabled)')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'none'],
                        help='Learning rate scheduler')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    """Get the appropriate device."""
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def create_model(args, num_classes: int = 8) -> nn.Module:
    """Create the appropriate model based on arguments."""
    common_kwargs = {
        'n_size': args.n_size,
        'num_classes': num_classes,
        'task': args.task,
    }
    
    # Add dropout if specified
    if args.dropout is not None:
        common_kwargs['dropout'] = args.dropout
    
    if args.model == 'diagonal_cnn':
        return DiagonalCNN(band_width=args.band_width, **common_kwargs)
    elif args.model == 'dense':
        return DenseNet(**common_kwargs)
    elif args.model == 'image_cnn':
        return ImageCNN(in_channels=1, **common_kwargs)
    elif args.model == 'image_resnet':
        return ImageResNet(in_channels=1, **common_kwargs)
    else:
        raise ValueError(f"Unknown model: {args.model}")


def get_input_type(model_name: str) -> str:
    """Map model name to input type."""
    if model_name == 'diagonal_cnn':
        return 'diagonal'
    elif model_name in ['image_cnn', 'image_resnet']:
        return 'image'
    else:
        return 'matrix'


def create_dataloaders_from_args(args) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders based on arguments."""
    input_type = get_input_type(args.model)
    
    if args.synthetic or args.data_dir is None:
        # Use synthetic data
        print("Using synthetic data generation...")
        print("  Note: Synthetic labels are approximations. For accurate labels,")
        print("        use png_builder2.py to generate data with solver profiling.")
        
        full_dataset = SyntheticBlockJacobiDataset(
            num_samples=args.num_samples,
            n_size=args.n_size,
            input_type=input_type,
            task=args.task,
            band_width=args.band_width,
            seed=args.seed,
        )
        
        # Split
        n_val = int(len(full_dataset) * args.val_split)
        n_train = len(full_dataset) - n_val
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader
    else:
        # Use real data
        return create_dataloaders(
            data_dir=args.data_dir,
            input_type=input_type,
            task=args.task,
            batch_size=args.batch_size,
            val_split=args.val_split,
            band_width=args.band_width,
        )


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    task: str,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        if task == 'classification':
            loss = criterion(output, target)
            pred = output.argmax(dim=1)
            correct = (pred == target).sum().item()
        else:
            loss = criterion(output, target)
            correct = 0  # Not applicable for regression
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        total_correct += correct
        total_samples += data.size(0)
    
    metrics = {
        'loss': total_loss / total_samples,
    }
    
    if task == 'classification':
        metrics['accuracy'] = total_correct / total_samples
    
    return metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            if task == 'classification':
                loss = criterion(output, target)
                pred = output.argmax(dim=1)
                correct = (pred == target).sum().item()
            else:
                loss = criterion(output, target)
                correct = 0
            
            total_loss += loss.item() * data.size(0)
            total_correct += correct
            total_samples += data.size(0)
            
            all_predictions.append(output.cpu())
            all_targets.append(target.cpu())
    
    metrics = {
        'loss': total_loss / total_samples,
    }
    
    if task == 'classification':
        metrics['accuracy'] = total_correct / total_samples
    else:
        # Compute MAE for regression (in normalized [0,1] space and original scale)
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)
        
        # MAE in normalized space
        mae_normalized = (predictions - targets).abs().mean().item()
        
        # MAE in original scale [0.05, 0.40]
        mae_original = mae_normalized * REGRESSION_RANGE
        
        metrics['mae'] = mae_original  # Report in original scale
        metrics['mae_normalized'] = mae_normalized
        
        # RMSE in original scale
        mse = ((predictions - targets) ** 2).mean().item()
        rmse_original = (mse ** 0.5) * REGRESSION_RANGE
        metrics['rmse'] = rmse_original
    
    return metrics


def train(args):
    """Main training function."""
    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.model}_{args.task}_{timestamp}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders_from_args(args)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    # Determine number of classes for classification
    num_classes = len(THRESHOLD_VALUES)  # 8 classes: [0.05, 0.10, ..., 0.40]
    
    # Create model
    model = create_model(args, num_classes=num_classes)
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}, Parameters: {n_params:,}")
    
    # Loss function
    if args.task == 'classification':
        # Label smoothing helps prevent overconfidence and improves generalization
        label_smoothing = getattr(args, 'label_smoothing', 0.1)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        criterion = nn.MSELoss()
    
    # Optimizer (Nadam = Adam + Nesterov momentum)
    optimizer = optim.NAdam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = None
    
    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_mae': [], 'val_rmse': [],  # For regression
    }
    
    print(f"\nStarting training for {args.epochs} epochs...")
    if args.early_stopping > 0:
        print(f"Early stopping enabled with patience={args.early_stopping}")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, args.task
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, args.task
        )
        
        # Update scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        
        if args.task == 'classification':
            history['train_acc'].append(train_metrics.get('accuracy', 0))
            history['val_acc'].append(val_metrics.get('accuracy', 0))
        else:
            history['val_mae'].append(val_metrics.get('mae', 0))
            history['val_rmse'].append(val_metrics.get('rmse', 0))
        
        # Print progress
        if args.task == 'classification':
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics.get('accuracy', 0):.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics.get('accuracy', 0):.4f}")
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics.get('mae', 0):.4f}")
        
        # Save best model and track improvement
        if args.save_best and val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, run_dir / 'best_model.pt')
            print(f"  -> Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
        
        # Early stopping check
        if args.early_stopping > 0 and epochs_without_improvement >= args.early_stopping:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {args.early_stopping} epochs)")
            break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, run_dir / 'final_model.pt')
    
    # Save history
    with open(run_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nResults saved to: {run_dir}")
    
    return model, history


if __name__ == '__main__':
    args = parse_args()
    train(args)
