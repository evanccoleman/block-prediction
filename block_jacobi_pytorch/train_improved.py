#!/usr/bin/env python
"""
Improved training script with:
- Early stopping
- Better regularization (label smoothing, increased dropout)
- Data augmentation for images
- Confusion matrix analysis
- Learning rate warmup
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from data import (
    BlockJacobiDataset,
    SyntheticBlockJacobiDataset,
    THRESHOLD_VALUES,
    create_dataloaders,
)
from models import DiagonalCNN, DenseNet, ImageCNN, ImageResNet


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing for better generalization."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(1)
        # Convert to one-hot
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        # Apply smoothing
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        # Cross entropy
        log_prob = torch.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=10, min_delta=0.001, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = None
        self.best_state = None
        self.should_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                
    def restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)


class ImageAugmentation:
    """Simple augmentation for matrix images."""
    def __init__(self, p_noise=0.3, p_dropout=0.2, noise_std=0.05):
        self.p_noise = p_noise
        self.p_dropout = p_dropout
        self.noise_std = noise_std
        
    def __call__(self, x):
        # Add Gaussian noise
        if torch.rand(1).item() < self.p_noise:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
            x = torch.clamp(x, 0, 1)
        
        # Random dropout of pixels
        if torch.rand(1).item() < self.p_dropout:
            mask = torch.rand_like(x) > 0.1  # Drop 10% of pixels
            x = x * mask.float()
            
        return x


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for better generalization."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss for mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def compute_confusion_matrix(model, loader, device, num_classes):
    """Compute confusion matrix for analysis."""
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion[t.long(), p.long()] += 1
                
    return confusion.numpy()


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           xlabel='Predicted', ylabel='True',
           title='Confusion Matrix')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=8)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def get_model_with_dropout(model_name, n_size, num_classes, dropout_rate=0.3):
    """Create model with specified dropout rate."""
    if model_name == 'diagonal_cnn':
        model = DiagonalCNN(n_size=n_size, num_classes=num_classes, task='classification')
        # DiagonalCNN already has dropout, but we can increase it
    elif model_name == 'dense':
        model = DenseNet(n_size=n_size, num_classes=num_classes, task='classification',
                        use_sparse_features=False, dropout=dropout_rate)
    elif model_name == 'image_cnn':
        model = ImageCNN(n_size=n_size, in_channels=1, num_classes=num_classes, 
                        task='classification', dropout=dropout_rate)
    elif model_name == 'image_resnet':
        model = ImageResNet(n_size=n_size, in_channels=1, num_classes=num_classes, 
                           task='classification', dropout=dropout_rate)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def train_improved(args):
    """Main training function with improvements."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine input type based on model
    input_type_map = {
        'diagonal_cnn': 'diagonal',
        'dense': 'matrix',
        'image_cnn': 'image',
        'image_resnet': 'image',
    }
    input_type = input_type_map[args.model]
    
    # Create dataset
    if args.data_dir:
        dataset = BlockJacobiDataset(
            data_dir=args.data_dir,
            input_type=input_type,
            task='classification',
            band_width=args.band_width,
        )
        n_size = 128  # Default, could be detected from data
    else:
        dataset = SyntheticBlockJacobiDataset(
            num_samples=args.num_samples,
            n_size=args.n_size,
            input_type=input_type,
            task='classification',
            band_width=args.band_width,
            seed=args.seed,
        )
        n_size = args.n_size
    
    # Split dataset
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {n_train}, Val samples: {n_val}")
    
    # Create model
    num_classes = len(THRESHOLD_VALUES)
    model = get_model_with_dropout(args.model, n_size, num_classes, args.dropout)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}, Parameters: {n_params:,}")
    
    # Loss function with label smoothing
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        print(f"Using label smoothing: {args.label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer with higher weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
    
    # Augmentation
    augment = ImageAugmentation() if args.augment and input_type in ['matrix', 'image'] else None
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.model}_improved_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for up to {args.epochs} epochs (early stopping patience={args.patience})...")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            # Apply augmentation
            if augment is not None:
                data = augment(data)
            
            optimizer.zero_grad()
            
            # Mixup
            if args.mixup > 0 and np.random.random() < 0.5:
                data, targets_a, targets_b, lam = mixup_data(data, target, args.mixup)
                output = model(data)
                loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)
            
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += target.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                val_correct += (pred == target).sum().item()
                val_total += target.size(0)
        
        # Metrics
        train_loss_avg = train_loss / train_total
        val_loss_avg = val_loss / val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Update scheduler
        if args.scheduler == 'plateau':
            scheduler.step(val_loss_avg)
        else:
            scheduler.step()
        
        # Save best model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss_avg:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss_avg:.4f}, Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.2e} <- BEST")
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss_avg:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss_avg:.4f}, Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.2e}")
        
        # Early stopping check
        early_stopping(val_loss_avg, model)
        if early_stopping.should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            early_stopping.restore(model)
            break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    
    # Compute confusion matrix
    print("\nComputing confusion matrix...")
    cm = compute_confusion_matrix(model, val_loader, device, num_classes)
    class_names = [f"{t:.2f}" for t in THRESHOLD_VALUES]
    plot_confusion_matrix(cm, class_names, output_dir / 'confusion_matrix.png')
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    for i, (name, acc) in enumerate(zip(class_names, per_class_acc)):
        print(f"  Class {name}: {acc:.4f} ({cm[i].sum()} samples)")
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Loss')
    
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_title('Accuracy')
    
    axes[2].plot(history['lr'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()
    
    # Save config
    config = {
        'model': args.model,
        'epochs_run': len(history['train_loss']),
        'best_val_loss': best_val_loss,
        'best_val_acc': max(history['val_acc']),
        'final_val_acc': history['val_acc'][-1],
        'label_smoothing': args.label_smoothing,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,
        'mixup': args.mixup,
        'augment': args.augment,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Improved Block-Jacobi Training')
    
    # Model
    parser.add_argument('--model', type=str, default='diagonal_cnn',
                       choices=['diagonal_cnn', 'dense', 'image_cnn', 'image_resnet'])
    
    # Data
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--num-samples', type=int, default=3000)
    parser.add_argument('--n-size', type=int, default=128)
    parser.add_argument('--band-width', type=int, default=10)
    parser.add_argument('--val-split', type=float, default=0.2)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine'])
    
    # Regularization (KEY IMPROVEMENTS)
    parser.add_argument('--weight-decay', type=float, default=0.05,
                       help='Weight decay (L2 regularization). Default: 0.05 (higher than before)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate. Default: 0.3')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing factor. Default: 0.1')
    parser.add_argument('--mixup', type=float, default=0.2,
                       help='Mixup alpha. 0 to disable. Default: 0.2')
    parser.add_argument('--augment', action='store_true',
                       help='Apply data augmentation')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping norm. Default: 1.0')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience. Default: 15')
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./experiments/improved')
    
    args = parser.parse_args()
    train_improved(args)


if __name__ == '__main__':
    main()
