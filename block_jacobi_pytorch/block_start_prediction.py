#!/usr/bin/env python
"""
Per-Entry Block Start Prediction (Original Paper Approach)

The original Götz & Anzt paper predicts, for each diagonal entry,
whether it is the START of a new block. This allows:
1. Variable block sizes across the matrix
2. Blocks that adapt to local structure
3. O(n) inference with O(1) model parameters

This is fundamentally different from predicting a single global
block size fraction (which is what we implemented initially).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


def visualize_block_start_prediction():
    """
    Visualize the difference between:
    1. Global block size prediction (our implementation)
    2. Per-entry block start prediction (original paper)
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Create a matrix with variable block structure
    n = 128
    np.random.seed(42)
    
    # Define variable blocks
    block_starts = [0, 15, 35, 50, 80, 95, 115]  # Variable block sizes!
    block_ends = block_starts[1:] + [n]
    
    # Generate matrix with this structure
    matrix = np.zeros((n, n))
    for start, end in zip(block_starts, block_ends):
        block_size = end - start
        # Dense block with some randomness
        block = np.random.randn(block_size, block_size) * 0.5
        block += np.eye(block_size) * 2  # Strong diagonal
        # Random sparsification
        mask = np.random.random((block_size, block_size)) < 0.7
        block *= mask
        matrix[start:end, start:end] = block
    
    # Add some off-diagonal noise
    noise = np.random.randn(n, n) * 0.1
    noise_mask = np.random.random((n, n)) < 0.02
    matrix += noise * noise_mask
    
    # Plot 1: Original matrix
    ax = axes[0, 0]
    ax.imshow(matrix != 0, cmap='gray_r', interpolation='nearest')
    ax.set_title('Matrix Sparsity Pattern\n(Variable Block Sizes)', fontsize=11)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Mark true block boundaries
    for bs in block_starts[1:]:
        ax.axhline(y=bs-0.5, color='red', linewidth=1, linestyle='--', alpha=0.7)
        ax.axvline(x=bs-0.5, color='red', linewidth=1, linestyle='--', alpha=0.7)
    
    # Plot 2: Global prediction approach (our implementation)
    ax = axes[0, 1]
    ax.imshow(matrix != 0, cmap='gray_r', interpolation='nearest')
    ax.set_title('Global Block Size Prediction\n(Single size for entire matrix)', fontsize=11)
    
    # Predict single block size (e.g., 20% = 25.6 ≈ 26)
    global_block_size = 26
    for i in range(0, n, global_block_size):
        ax.axhline(y=i-0.5, color='blue', linewidth=2, alpha=0.7)
        ax.axvline(x=i-0.5, color='blue', linewidth=2, alpha=0.7)
    ax.set_xlabel('Predicted blocks (uniform size)')
    
    # Plot 3: Per-entry prediction approach (original paper)
    ax = axes[0, 2]
    ax.imshow(matrix != 0, cmap='gray_r', interpolation='nearest')
    ax.set_title('Per-Entry Block Start Prediction\n(Variable sizes, adapts to structure)', fontsize=11)
    
    # Show predicted block starts (assume perfect prediction for visualization)
    for bs in block_starts:
        ax.axhline(y=bs-0.5, color='green', linewidth=2, alpha=0.7)
        ax.axvline(x=bs-0.5, color='green', linewidth=2, alpha=0.7)
    ax.set_xlabel('Predicted blocks (variable size)')
    
    # Plot 4: Diagonal band extraction
    ax = axes[1, 0]
    band_width = 10
    band = np.zeros((2*band_width+1, n))
    for i in range(n):
        for k in range(-band_width, band_width+1):
            j = i + k
            if 0 <= j < n:
                band[k + band_width, i] = matrix[i, j]
    
    ax.imshow(band != 0, cmap='gray_r', interpolation='nearest', aspect='auto')
    ax.set_title('Diagonal Band Extraction\n(Input to DiagonalCNN)', fontsize=11)
    ax.set_xlabel('Diagonal position (i)')
    ax.set_ylabel('Band offset (k)')
    
    # Mark block starts on the band
    for bs in block_starts[1:]:
        ax.axvline(x=bs, color='red', linewidth=1, linestyle='--', alpha=0.7)
    
    # Plot 5: Sliding window prediction
    ax = axes[1, 1]
    
    # Show sliding window concept
    window_size = 21  # 2k+1
    positions = [20, 35, 50, 80]
    colors = ['blue', 'green', 'orange', 'purple']
    
    ax.imshow(band != 0, cmap='gray_r', interpolation='nearest', aspect='auto', alpha=0.3)
    
    for pos, color in zip(positions, colors):
        # Draw window
        rect = plt.Rectangle((pos - window_size//2, -0.5), window_size, 2*band_width+1,
                            linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Mark if this is a block start
        is_start = pos in block_starts
        marker = '★' if is_start else '○'
        ax.text(pos, -2, marker, ha='center', va='center', fontsize=14, color=color)
    
    ax.set_title('Sliding Window Prediction\n(Each position: block start?)', fontsize=11)
    ax.set_xlabel('Diagonal position')
    ax.set_ylim(-3, 2*band_width+1)
    
    # Plot 6: Output comparison
    ax = axes[1, 2]
    
    # Binary predictions along diagonal
    x = np.arange(n)
    y_true = np.zeros(n)
    y_true[block_starts] = 1
    
    ax.stem(x, y_true, linefmt='g-', markerfmt='go', basefmt='k-', label='True block starts')
    ax.set_xlabel('Diagonal position')
    ax.set_ylabel('Block start prediction')
    ax.set_title('Per-Entry Binary Predictions\n(1 = start of new block)', fontsize=11)
    ax.set_ylim(-0.1, 1.2)
    ax.legend()
    
    # Add block size annotations
    for i, (start, end) in enumerate(zip(block_starts, block_ends)):
        mid = (start + end) / 2
        size = end - start
        ax.annotate(f'size={size}', xy=(mid, 0.5), ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig('block_start_prediction_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Saved: block_start_prediction_comparison.png")
    
    # Summary statistics
    print("\n" + "="*70)
    print("COMPARISON: Global vs Per-Entry Prediction")
    print("="*70)
    print(f"\nTrue block structure:")
    print(f"  Block starts: {block_starts}")
    print(f"  Block sizes:  {[e-s for s,e in zip(block_starts, block_ends)]}")
    print(f"  Mean size: {np.mean([e-s for s,e in zip(block_starts, block_ends)]):.1f}")
    print(f"  Size variance: {np.var([e-s for s,e in zip(block_starts, block_ends)]):.1f}")
    
    print(f"\nGlobal prediction (single block size = {global_block_size}):")
    predicted_starts_global = list(range(0, n, global_block_size))
    print(f"  Predicted starts: {predicted_starts_global}")
    
    # Compute alignment error
    errors_global = []
    for true_start in block_starts:
        nearest = min(predicted_starts_global, key=lambda x: abs(x - true_start))
        errors_global.append(abs(nearest - true_start))
    print(f"  Mean alignment error: {np.mean(errors_global):.1f} entries")
    
    print(f"\nPer-entry prediction (if perfect):")
    print(f"  Predicted starts: {block_starts}")
    print(f"  Mean alignment error: 0.0 entries")
    
    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("="*70)
    print("""
Per-entry prediction is more powerful because:
1. It can capture VARIABLE block sizes
2. It adapts to LOCAL matrix structure
3. It's the same complexity: O(n) predictions, each O(1)

The trade-off:
- Global: Simpler labels (1 per matrix), but assumes uniform blocks
- Per-entry: Richer labels (n per matrix), captures heterogeneity

For matrices with consistent block structure, both work.
For matrices with variable blocks, per-entry is essential.
""")


class BlockStartCNN(nn.Module):
    """
    Per-entry block start prediction network.
    
    For each position i on the diagonal, predicts P(block starts at i).
    
    Uses a sliding window approach:
    - Extract local patch around position i
    - Predict binary: start (1) or continue (0)
    
    This is closer to the original Götz & Anzt architecture.
    """
    
    def __init__(self, band_width: int = 10, window_size: int = 21):
        super().__init__()
        self.band_width = band_width
        self.window_size = window_size
        self.band_height = 2 * band_width + 1  # 21 for k=10
        
        # Input: (batch, 1, band_height, window_size)
        # e.g., (batch, 1, 21, 21)
        
        # Convolutional feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 21x21 -> 10x10
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 10x10 -> 5x5
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),  # -> 2x2
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),  # Binary output
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1, band_height, window_size) local patches
            
        Returns:
            (batch, 1) logits for block start probability
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def predict_sequence(self, diagonal_band: torch.Tensor) -> torch.Tensor:
        """
        Predict block starts for entire diagonal.
        
        Args:
            diagonal_band: (1, 1, band_height, n) full diagonal band
            
        Returns:
            (n,) binary predictions for each position
        """
        n = diagonal_band.shape[-1]
        half_window = self.window_size // 2
        
        # Pad the band
        padded = F.pad(diagonal_band, (half_window, half_window), mode='constant', value=0)
        
        predictions = []
        self.eval()
        
        with torch.no_grad():
            for i in range(n):
                # Extract window centered at position i
                window = padded[:, :, :, i:i+self.window_size]
                logit = self(window)
                prob = torch.sigmoid(logit)
                predictions.append(prob.item())
        
        return torch.tensor(predictions)


def extract_patches_for_training(diagonal_band: np.ndarray, 
                                  block_starts: list,
                                  window_size: int = 21) -> tuple:
    """
    Extract training patches from a diagonal band.
    
    Args:
        diagonal_band: (band_height, n) array
        block_starts: list of positions where blocks start
        window_size: size of sliding window
        
    Returns:
        patches: (n_patches, 1, band_height, window_size)
        labels: (n_patches,) binary labels
    """
    band_height, n = diagonal_band.shape
    half_window = window_size // 2
    
    # Pad
    padded = np.pad(diagonal_band, ((0, 0), (half_window, half_window)), mode='constant')
    
    patches = []
    labels = []
    
    for i in range(n):
        window = padded[:, i:i+window_size]
        patches.append(window)
        labels.append(1.0 if i in block_starts else 0.0)
    
    patches = np.array(patches)[:, np.newaxis, :, :]  # Add channel dim
    labels = np.array(labels)
    
    return patches, labels


if __name__ == '__main__':
    # Generate visualization
    visualize_block_start_prediction()
    
    # Test the model
    print("\n" + "="*70)
    print("Testing BlockStartCNN")
    print("="*70)
    
    model = BlockStartCNN(band_width=10, window_size=21)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 1, 21, 21)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test sequence prediction
    print("\nTesting sequence prediction...")
    diagonal_band = torch.randn(1, 1, 21, 128)
    predictions = model.predict_sequence(diagonal_band)
    print(f"Diagonal band shape: {diagonal_band.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
