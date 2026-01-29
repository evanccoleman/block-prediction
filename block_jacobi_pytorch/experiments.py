#!/usr/bin/env python
"""
Experiment script to compare all three model architectures.

Runs training for:
1. DiagonalCNN (Götz & Anzt approach)
2. DenseNet (raw matrix)
3. ImageResNet (PNG image)

And compares their performance on synthetic data.
"""

import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import DiagonalCNN, DenseNet, ImageResNet
from data import SyntheticBlockJacobiDataset, THRESHOLD_VALUES

# Number of classes (8 threshold values)
NUM_CLASSES = len(THRESHOLD_VALUES)


def run_experiment(
    model_class: type,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_size: int = 128,
    num_epochs: int = 50,
    lr: float = 1e-4,
) -> Dict:
    """Run a single experiment with a model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Create model with 8 output classes
    if model_name == 'DiagonalCNN':
        model = model_class(n_size=n_size, num_classes=NUM_CLASSES, task='classification')
    elif model_name == 'DenseNet':
        model = model_class(n_size=n_size, num_classes=NUM_CLASSES, task='classification', 
                           use_sparse_features=False)  # Disable for fair comparison
    else:
        model = model_class(n_size=n_size, in_channels=1, num_classes=NUM_CLASSES, task='classification')
    
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    
    # Loss and optimizer - CrossEntropyLoss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.NAdam(model.parameters(), lr=lr, weight_decay=0.02)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
            
            # Accuracy
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += target.size(0)
        
        # Validate
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
        
        # Calculate metrics
        train_loss_avg = train_loss / len(train_loader.dataset)
        val_loss_avg = val_loss / len(val_loader.dataset)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step(val_loss_avg)
        
        # Record
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss_avg:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss_avg:.4f}, Acc: {val_acc:.4f}")
    
    total_time = time.time() - start_time
    
    # Final metrics
    results = {
        'model_name': model_name,
        'n_params': n_params,
        'best_val_loss': best_val_loss,
        'final_val_acc': history['val_acc'][-1],
        'best_val_acc': max(history['val_acc']),
        'training_time': total_time,
        'history': history,
    }
    
    print(f"\nFinal: Val Acc = {results['final_val_acc']:.4f}")
    print(f"Best Val Acc: {results['best_val_acc']:.4f}")
    print(f"Training time: {total_time:.1f}s")
    
    return results


def plot_results(all_results: List[Dict], output_dir: Path):
    """Plot comparison of all models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = {'DiagonalCNN': 'blue', 'DenseNet': 'green', 'ImageResNet': 'red'}
    
    # Loss curves
    ax = axes[0, 0]
    for result in all_results:
        name = result['model_name']
        ax.plot(result['history']['train_loss'], '--', color=colors[name], alpha=0.5)
        ax.plot(result['history']['val_loss'], '-', color=colors[name], label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax = axes[0, 1]
    for result in all_results:
        name = result['model_name']
        ax.plot(result['history']['train_acc'], '--', color=colors[name], alpha=0.5)
        ax.plot(result['history']['val_acc'], '-', color=colors[name], label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Parameter count and training time comparison
    ax = axes[1, 0]
    model_names = [r['model_name'] for r in all_results]
    x = np.arange(len(model_names))
    params = [r['n_params'] / 1e6 for r in all_results]  # In millions
    times = [r['training_time'] for r in all_results]
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x - 0.2, params, 0.4, label='Parameters (M)', color='steelblue')
    bars2 = ax2.bar(x + 0.2, times, 0.4, label='Train Time (s)', color='coral')
    
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Parameters (millions)', color='steelblue')
    ax2.set_ylabel('Training Time (s)', color='coral')
    ax.set_title('Model Complexity')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Summary bar chart
    ax = axes[1, 1]
    model_names = [r['model_name'] for r in all_results]
    x = np.arange(len(model_names))
    width = 0.35
    
    final_acc = [r['final_val_acc'] for r in all_results]
    best_acc = [r['best_val_acc'] for r in all_results]
    
    ax.bar(x - width/2, final_acc, width, label='Final Val Accuracy', color='skyblue')
    ax.bar(x + width/2, best_acc, width, label='Best Val Accuracy', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Accuracy')
    ax.set_title('Final Performance Comparison')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=150)
    plt.close()
    print(f"Saved comparison plot to {output_dir / 'comparison.png'}")


def main():
    # Configuration
    n_samples = 1000  # Reduced for faster experimentation
    n_size = 128
    batch_size = 16
    num_epochs = 50
    seed = 42
    
    output_dir = Path('./experiment_results')
    output_dir.mkdir(exist_ok=True)
    
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets for each input type
    print("\nGenerating synthetic data...")
    
    # DiagonalCNN dataset
    diagonal_dataset = SyntheticBlockJacobiDataset(
        num_samples=n_samples, n_size=n_size, input_type='diagonal',
        task='classification', seed=seed
    )
    
    # DenseNet/ImageResNet dataset (same data, different format)
    matrix_dataset = SyntheticBlockJacobiDataset(
        num_samples=n_samples, n_size=n_size, input_type='matrix',
        task='classification', seed=seed
    )
    
    # Split datasets
    n_val = int(n_samples * 0.2)
    n_train = n_samples - n_val
    
    # Diagonal
    diag_train, diag_val = torch.utils.data.random_split(
        diagonal_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    diag_train_loader = DataLoader(diag_train, batch_size=batch_size, shuffle=True)
    diag_val_loader = DataLoader(diag_val, batch_size=batch_size, shuffle=False)
    
    # Matrix (for both Dense and Image models)
    mat_train, mat_val = torch.utils.data.random_split(
        matrix_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    mat_train_loader = DataLoader(mat_train, batch_size=batch_size, shuffle=True)
    mat_val_loader = DataLoader(mat_val, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {n_train}, Val samples: {n_val}")
    
    # Run experiments
    all_results = []
    
    # 1. DiagonalCNN
    results = run_experiment(
        DiagonalCNN, 'DiagonalCNN',
        diag_train_loader, diag_val_loader,
        device, n_size, num_epochs
    )
    all_results.append(results)
    
    # 2. DenseNet
    results = run_experiment(
        DenseNet, 'DenseNet',
        mat_train_loader, mat_val_loader,
        device, n_size, num_epochs
    )
    all_results.append(results)
    
    # 3. ImageResNet
    results = run_experiment(
        ImageResNet, 'ImageResNet',
        mat_train_loader, mat_val_loader,
        device, n_size, num_epochs
    )
    all_results.append(results)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'Params':>12} {'Final Acc':>10} {'Best Acc':>10} {'Time (s)':>10}")
    print("-"*60)
    for r in all_results:
        print(f"{r['model_name']:<15} {r['n_params']:>12,} {r['final_val_acc']:>10.4f} "
              f"{r['best_val_acc']:>10.4f} {r['training_time']:>10.1f}")
    
    # Save results
    summary = {
        'config': {
            'n_samples': n_samples,
            'n_size': n_size,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'seed': seed,
        },
        'results': [{k: v for k, v in r.items() if k != 'history'} for r in all_results],
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Plot
    plot_results(all_results, output_dir)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
