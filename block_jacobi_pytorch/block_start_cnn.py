#!/usr/bin/env python
"""
Block Start Prediction CNN (Original Paper Approach)

This implements the per-entry block start prediction from:
Götz & Anzt, "Machine Learning-Aided Numerical Linear Algebra: 
Convolutional Neural Networks for the Efficient Preconditioner Generation"

Key differences from our global prediction approach:
1. Predicts binary (block start / continue) for EACH diagonal position
2. Uses sliding window along diagonal band  
3. Allows variable block sizes across the matrix
4. Model parameters are O(1), inference is O(n)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class BlockStartCNN(nn.Module):
    """
    Per-entry block start prediction network.
    
    Architecture based on Götz & Anzt paper:
    - Input: Local window of diagonal band centered at position i
    - Output: P(block starts at position i)
    
    The key insight is that block boundaries are LOCAL features -
    you can detect them by looking at a small neighborhood.
    
    Input shape: (batch, 1, band_height, window_width)
                 e.g., (batch, 1, 21, 21) for k=10, window=21
    
    Output shape: (batch, 1) - logit for block start probability
    """
    
    def __init__(self, band_width: int = 10, window_size: int = 21,
                 base_channels: int = 32, dropout: float = 0.5):
        """
        Args:
            band_width: Half-width of diagonal band (k), so band height = 2k+1
            window_size: Width of sliding window (should be odd)
            base_channels: Number of channels in first conv layer
            dropout: Dropout rate for classifier
        """
        super().__init__()
        
        self.band_width = band_width
        self.window_size = window_size
        self.band_height = 2 * band_width + 1
        
        # Ensure window size is odd
        if window_size % 2 == 0:
            self.window_size = window_size + 1
        
        # ============================================
        # Part 1: Residual Denoising (from paper)
        # ============================================
        # Two conv layers with residual connection to reduce noise
        self.denoise_conv1 = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)
        self.denoise_bn1 = nn.BatchNorm2d(base_channels)
        self.denoise_conv2 = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.denoise_bn2 = nn.BatchNorm2d(1)
        
        # ============================================
        # Part 2: Feature Extraction
        # ============================================
        # Convolutional layers to detect block boundary patterns
        self.features = nn.Sequential(
            # Block 1: Detect local edges/transitions
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: Larger receptive field
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: Global context
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        # Adaptive pooling to fixed size (handles variable window sizes)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # ============================================
        # Part 3: Binary Classifier
        # ============================================
        classifier_input = base_channels * 4 * 2 * 2
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(classifier_input, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # Single output: block start logit
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of local windows.
        
        Args:
            x: (batch, 1, band_height, window_size) tensor
            
        Returns:
            (batch, 1) logits for block start probability
        """
        # Part 1: Residual denoising
        identity = x
        out = F.selu(self.denoise_bn1(self.denoise_conv1(x)))
        out = self.denoise_bn2(self.denoise_conv2(out))
        out = out + identity  # Residual connection
        
        # Part 2: Feature extraction
        out = self.features(out)
        out = self.adaptive_pool(out)
        
        # Part 3: Classification
        out = self.classifier(out)
        
        return out
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability instead of logit."""
        return torch.sigmoid(self.forward(x))
    
    def predict_full_diagonal(self, diagonal_band: torch.Tensor, 
                               threshold: float = 0.5,
                               batch_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict block starts for an entire diagonal band.
        
        This is the main inference function - slides the window across
        the entire diagonal and predicts block start probability at each position.
        
        Args:
            diagonal_band: (1, band_height, n) or (band_height, n) tensor
            threshold: Probability threshold for block start
            batch_size: Batch size for efficient inference
            
        Returns:
            probabilities: (n,) tensor of block start probabilities
            block_starts: (n,) binary tensor of predicted block starts
        """
        self.eval()
        
        # Handle input dimensions
        if diagonal_band.dim() == 2:
            diagonal_band = diagonal_band.unsqueeze(0)  # Add channel dim
        if diagonal_band.dim() == 3:
            diagonal_band = diagonal_band.unsqueeze(0)  # Add batch dim
        
        # diagonal_band is now (1, 1, band_height, n)
        n = diagonal_band.shape[-1]
        half_window = self.window_size // 2
        
        # Pad the band for edge handling
        padded = F.pad(diagonal_band, (half_window, half_window), mode='replicate')
        
        # Extract all windows
        windows = []
        for i in range(n):
            window = padded[:, :, :, i:i + self.window_size]
            windows.append(window)
        
        # Stack into batches for efficient inference
        all_windows = torch.cat(windows, dim=0)  # (n, 1, band_height, window_size)
        
        # Predict in batches
        probabilities = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = all_windows[i:i + batch_size]
                logits = self.forward(batch)
                probs = torch.sigmoid(logits)
                probabilities.append(probs)
        
        probabilities = torch.cat(probabilities, dim=0).squeeze()  # (n,)
        block_starts = (probabilities > threshold).float()
        
        # First position is always a block start
        block_starts[0] = 1.0
        
        return probabilities, block_starts
    
    def extract_block_sizes(self, block_starts: torch.Tensor) -> List[int]:
        """
        Convert binary block start predictions to block sizes.
        
        Args:
            block_starts: (n,) binary tensor
            
        Returns:
            List of block sizes
        """
        starts = torch.where(block_starts > 0.5)[0].tolist()
        if 0 not in starts:
            starts = [0] + starts
        
        n = len(block_starts)
        sizes = []
        for i in range(len(starts)):
            if i + 1 < len(starts):
                sizes.append(starts[i + 1] - starts[i])
            else:
                sizes.append(n - starts[i])
        
        return sizes


class BlockStartDataset(Dataset):
    """
    Dataset for per-entry block start prediction.
    
    Extracts sliding windows from diagonal bands and provides
    binary labels for each position.
    """
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 band_width: int = 10,
                 window_size: int = 21,
                 samples_per_matrix: Optional[int] = None,
                 balance_classes: bool = True):
        """
        Args:
            data_dir: Directory containing matrix data
            band_width: Half-width of diagonal band
            window_size: Size of sliding window
            samples_per_matrix: If set, subsample this many positions per matrix
            balance_classes: If True, balance positive/negative samples
        """
        self.data_dir = Path(data_dir)
        self.band_width = band_width
        self.window_size = window_size
        self.samples_per_matrix = samples_per_matrix
        self.balance_classes = balance_classes
        self.band_height = 2 * band_width + 1
        
        # Find all samples
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        """Load all windows and labels from the dataset."""
        samples = []
        
        # Check directory structure
        metadata_dir = self.data_dir / 'metadata'
        matrices_dir = self.data_dir / 'matrices'
        
        if metadata_dir.exists():
            # Subfolder structure
            json_files = sorted(metadata_dir.glob('*.json'))
        else:
            # Flat structure
            json_files = sorted(self.data_dir.glob('*.json'))
        
        print(f"Loading {len(json_files)} matrices for block start prediction...")
        
        for json_path in json_files:
            # Load metadata
            with open(json_path) as f:
                metadata = json.load(f)
            
            # Get matrix path
            stem = json_path.stem
            if matrices_dir.exists():
                npz_path = matrices_dir / f"{stem}.npz"
            else:
                npz_path = self.data_dir / f"{stem}.npz"
            
            if not npz_path.exists():
                continue
            
            # Load matrix and extract diagonal band
            data = np.load(npz_path)
            from scipy import sparse
            matrix = sparse.csr_matrix(
                (data['data'], data['indices'], data['indptr']),
                shape=tuple(data['shape'])
            )
            
            n = matrix.shape[0]
            diagonal_band = self._extract_diagonal_band(matrix)
            
            # Get block start labels
            # This requires the metadata to contain block structure info
            # For now, we'll use a heuristic based on the optimal threshold
            block_starts = self._get_block_starts(metadata, n)
            
            # Extract windows
            half_window = self.window_size // 2
            padded_band = np.pad(diagonal_band, 
                                ((0, 0), (half_window, half_window)), 
                                mode='edge')
            
            # Decide which positions to sample
            if self.samples_per_matrix is not None:
                if self.balance_classes:
                    # Sample equal positives and negatives
                    positive_idx = np.where(block_starts)[0]
                    negative_idx = np.where(~block_starts)[0]
                    
                    n_pos = min(len(positive_idx), self.samples_per_matrix // 2)
                    n_neg = min(len(negative_idx), self.samples_per_matrix // 2)
                    
                    if n_pos > 0:
                        pos_sample = np.random.choice(positive_idx, n_pos, replace=False)
                    else:
                        pos_sample = np.array([], dtype=int)
                    neg_sample = np.random.choice(negative_idx, n_neg, replace=False)
                    
                    positions = np.concatenate([pos_sample, neg_sample])
                else:
                    positions = np.random.choice(n, self.samples_per_matrix, replace=False)
            else:
                positions = np.arange(n)
            
            for pos in positions:
                window = padded_band[:, pos:pos + self.window_size]
                label = float(block_starts[pos])
                samples.append({
                    'window': window.astype(np.float32),
                    'label': label,
                    'position': pos,
                    'matrix_id': stem,
                })
        
        print(f"Loaded {len(samples)} training windows")
        
        # Report class balance
        labels = [s['label'] for s in samples]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        print(f"  Positive (block starts): {n_pos} ({100*n_pos/len(labels):.1f}%)")
        print(f"  Negative (continues): {n_neg} ({100*n_neg/len(labels):.1f}%)")
        
        return samples
    
    def _extract_diagonal_band(self, matrix) -> np.ndarray:
        """Extract diagonal band from sparse matrix."""
        n = matrix.shape[0]
        band = np.zeros((self.band_height, n), dtype=np.float32)
        
        # Convert to dense for band extraction (could optimize for large matrices)
        dense = matrix.toarray()
        
        for i in range(n):
            for k_idx, k in enumerate(range(-self.band_width, self.band_width + 1)):
                j = i + k
                if 0 <= j < n:
                    band[k_idx, i] = dense[i, j]
        
        # Normalize: convert to binary (sparsity pattern)
        band = (band != 0).astype(np.float32)
        
        return band
    
    def _get_block_starts(self, metadata: Dict, n: int) -> np.ndarray:
        """
        Determine block start positions from metadata.
        
        This is a key function - in the original paper, block starts
        were determined by the actual block-Jacobi structure.
        
        For our synthetic data, we can infer from the optimal block size.
        """
        block_starts = np.zeros(n, dtype=bool)
        block_starts[0] = True  # First position always starts a block
        
        # Try to get from metadata
        if 'block_starts' in metadata:
            # Direct block start positions
            for pos in metadata['block_starts']:
                if pos < n:
                    block_starts[pos] = True
        elif 'block_size' in metadata:
            # Uniform block size
            block_size = metadata['block_size']
            for i in range(0, n, block_size):
                block_starts[i] = True
        elif 'labels' in metadata:
            # Infer from optimal threshold
            labels = metadata['labels']
            if 'class_optimal_time' in labels:
                threshold = labels['class_optimal_time']
            elif 'class_optimal_iterations' in labels:
                threshold = labels['class_optimal_iterations']
            else:
                threshold = 0.1  # Default
            
            # Convert threshold to block size
            block_size = max(1, int(threshold * n))
            for i in range(0, n, block_size):
                block_starts[i] = True
        else:
            # Fallback: use 10% block size
            block_size = max(1, n // 10)
            for i in range(0, n, block_size):
                block_starts[i] = True
        
        return block_starts
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        window = torch.tensor(sample['window'], dtype=torch.float32)
        window = window.unsqueeze(0)  # Add channel dimension: (1, band_height, window_size)
        
        label = torch.tensor([sample['label']], dtype=torch.float32)
        
        return window, label


def train_block_start_model(
    data_dir: str,
    output_dir: str = './experiments/block_start',
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    band_width: int = 10,
    window_size: int = 21,
    samples_per_matrix: int = 200,
    device: str = 'auto'
):
    """
    Train the BlockStartCNN model.
    
    Args:
        data_dir: Directory containing matrix data
        output_dir: Directory to save results
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        band_width: Diagonal band half-width
        window_size: Sliding window size
        samples_per_matrix: Number of samples per matrix (for efficiency)
        device: 'cuda', 'cpu', or 'auto'
    """
    import time
    from datetime import datetime
    
    # Setup
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    print("\nLoading dataset...")
    full_dataset = BlockStartDataset(
        data_dir=data_dir,
        band_width=band_width,
        window_size=window_size,
        samples_per_matrix=samples_per_matrix,
        balance_classes=True
    )
    
    # Split into train/val
    n_val = int(len(full_dataset) * 0.2)
    n_train = len(full_dataset) - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {n_train}, Val samples: {n_val}")
    
    # Compute class imbalance for weighting
    all_labels = [s['label'] for s in full_dataset.samples]
    n_positive = sum(all_labels)
    n_negative = len(all_labels) - n_positive
    
    # pos_weight: how much more to weight positive samples
    # With 5% positive, this is ~19x
    pos_weight = torch.tensor([n_negative / (n_positive + 1e-10)]).to(device)
    print(f"Class imbalance: {n_positive:.0f} positive, {n_negative:.0f} negative")
    print(f"Using pos_weight={pos_weight.item():.2f} to balance classes")
    
    # Create model
    model = BlockStartCNN(
        band_width=band_width,
        window_size=window_size,
        dropout=0.5
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Loss and optimizer
    # Use weighted BCE to handle class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5  # Monitor F1, not loss
    )
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_val_f1 = 0.0  # Track F1 instead of loss for imbalanced data
    
    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for windows, labels in train_loader:
            windows, labels = windows.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(windows)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * windows.size(0)
        
        train_loss /= n_train
        
        # Validate
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for windows, labels in val_loader:
                windows, labels = windows.to(device), labels.to(device)
                outputs = model(windows)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * windows.size(0)
                
                probs = torch.sigmoid(outputs)
                # Use lower threshold (0.3) since positives are rare and valuable
                preds = (probs > 0.3).float()
                all_probs.extend(probs.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        val_loss /= n_val
        
        # Metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = (all_preds == all_labels).mean()
        
        # F1 score, precision, recall
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(accuracy)
        history['val_f1'].append(f1)
        
        # Update scheduler (monitoring F1)
        scheduler.step(f1)
        
        # Save best model (based on F1, not loss)
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Prec: {precision:.3f}, Rec: {recall:.3f}, F1: {f1:.4f} <- BEST")
        else:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Prec: {precision:.3f}, Rec: {recall:.3f}, F1: {f1:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Best validation F1: {best_val_f1:.4f}")
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config = {
        'band_width': band_width,
        'window_size': window_size,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'best_val_f1': best_val_f1,
        'n_params': n_params,
        'pos_weight': pos_weight.item(),
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    return model, history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BlockStartCNN')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing matrix data')
    parser.add_argument('--output-dir', type=str, default='./experiments/block_start',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--band-width', type=int, default=10)
    parser.add_argument('--window-size', type=int, default=21)
    parser.add_argument('--samples-per-matrix', type=int, default=200)
    
    args = parser.parse_args()
    
    train_block_start_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        band_width=args.band_width,
        window_size=args.window_size,
        samples_per_matrix=args.samples_per_matrix,
    )
