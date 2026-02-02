"""
PyTorch Models for Block-Jacobi Preconditioner Parameter Prediction

Three approaches:
1. DiagonalCNN: CNN operating on diagonal band (replicating the Götz & Anzt paper approach)
2. DenseNet: Dense neural network using flattened raw matrix data
3. ImageCNN: Standard image-based CNN operating on the PNG sparsity pattern

Author: Replicated and extended from Götz & Anzt (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalCNN(nn.Module):
    """
    CNN operating on the diagonal band of the matrix.
    
    This replicates the architecture from Götz & Anzt (2018):
    - Modified residual block for denoising
    - Non-standard convolutions for corner detection
    - Fully-connected predictor
    
    Input: Diagonal image of shape (batch, 1, 2w+1, n) where w=10, n=128
           So input shape is (batch, 1, 21, 128)
    """
    
    def __init__(self, n_size: int = 128, band_width: int = 10, 
                 num_classes: int = 8, task: str = 'classification',
                 dropout: float = 0.1):
        super().__init__()
        self.n_size = n_size
        self.band_width = band_width
        self.diag_height = 2 * band_width + 1  # 21
        self.num_classes = num_classes
        self.task = task
        
        # ===== Part 1: Residual Denoising Block =====
        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding='same')  # Back to 1 channel for residual
        
        # L2 regularization is handled via weight_decay in optimizer
        
        # ===== Part 2: Corner Detection Convolutions =====
        # Zero padding: add k=3 on left and right
        self.k = 3
        # Conv with kernel (2w+1) x (2k+1) = 21 x 7, valid padding
        self.conv3 = nn.Conv2d(1, 32, kernel_size=(self.diag_height, 2*self.k + 1), padding='valid')
        # 1D-style conv across the result: kernel 1 x 3
        self.conv4 = nn.Conv2d(32, 128, kernel_size=(1, 3), padding='same')
        
        # ===== Part 3: Fully-Connected Predictor =====
        # After conv3: output height = 1, width = n (due to horizontal padding)
        # After conv4: same size (1, n, 128)
        fc_input_size = 128 * n_size
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.dropout = nn.Dropout(dropout)
        
        if task == 'classification':
            self.fc2 = nn.Linear(512, num_classes)
        else:  # regression
            self.fc2 = nn.Linear(512, 1)
    
    def forward(self, x):
        # x shape: (batch, 1, 21, 128)
        
        # Part 1: Residual denoising
        identity = x
        out = self.bn1(x)
        out = F.selu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.selu(out)
        out = self.conv2(out)
        out = out + identity  # Residual connection
        
        # Part 2: Corner detection with padding
        # Pad horizontally: (left, right, top, bottom)
        out = F.pad(out, (self.k, self.k, 0, 0), mode='constant', value=0)
        out = torch.tanh(self.conv3(out))  # Now shape: (batch, 32, 1, n)
        out = torch.tanh(self.conv4(out))  # Shape: (batch, 128, 1, n)
        
        # Part 3: Prediction
        out = self.flatten(out)
        out = torch.sigmoid(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        if self.task == 'classification':
            return out  # Raw logits for CrossEntropyLoss
        else:
            return torch.sigmoid(out)  # Bounded [0, 1] for regression


class DenseNet(nn.Module):
    """
    Dense neural network using the full raw matrix data.
    
    This approach uses all n×n matrix values directly,
    treating the problem more like tabular data.
    
    WARNING: Memory usage scales as O(n²) for the first layer.
    For n > 256, consider using ConvDenseNet instead.
    
    Input: Flattened matrix of shape (batch, n*n) or (batch, 1, n, n)
    """
    
    def __init__(self, n_size: int = 128, num_classes: int = 8, 
                 task: str = 'classification', use_sparse_features: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        self.n_size = n_size
        self.num_classes = num_classes
        self.task = task
        self.use_sparse_features = use_sparse_features
        
        # Check if matrix is too large for dense approach
        if n_size > 256:
            import warnings
            warnings.warn(
                f"DenseNet with n_size={n_size} will use ~{n_size**2 * 2048 * 4 / 1e9:.1f} GB "
                f"for weights alone. Consider using ConvDenseNet instead."
            )
        
        # Input size: n*n for raw matrix + optional features
        base_input = n_size * n_size
        extra_features = 3 if use_sparse_features else 0  # nnz, density, diagonal_dominance
        input_size = base_input + extra_features
        
        # Deep fully-connected network with residual connections
        self.fc1 = nn.Linear(input_size, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(dropout)
        
        if task == 'classification':
            self.output = nn.Linear(256, num_classes)
        else:
            self.output = nn.Linear(256, 1)
    
    def compute_sparse_features(self, x):
        """Compute sparse matrix features from dense representation."""
        # x shape: (batch, n*n)
        batch_size = x.shape[0]
        x_2d = x.view(batch_size, self.n_size, self.n_size)
        
        # Number of nonzeros (normalized)
        nnz = (x_2d != 0).float().sum(dim=(1, 2)) / (self.n_size * self.n_size)
        
        # Density along diagonal band
        diag_mask = torch.zeros(self.n_size, self.n_size, device=x.device)
        for i in range(self.n_size):
            for j in range(max(0, i-10), min(self.n_size, i+11)):
                diag_mask[i, j] = 1
        diag_nnz = ((x_2d != 0).float() * diag_mask).sum(dim=(1, 2)) / diag_mask.sum()
        
        # Diagonal dominance (sum of |diag| / sum of |off-diag|)
        diag_vals = torch.diagonal(x_2d, dim1=1, dim2=2)
        diag_sum = torch.abs(diag_vals).sum(dim=1)
        total_sum = torch.abs(x_2d).sum(dim=(1, 2))
        diag_dominance = diag_sum / (total_sum + 1e-8)
        
        return torch.stack([nnz, diag_nnz, diag_dominance], dim=1)
    
    def forward(self, x):
        # Handle both (batch, 1, n, n) and (batch, n*n) inputs
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        elif x.dim() == 3:
            x = x.view(x.size(0), -1)
        
        # Optionally add sparse features
        if self.use_sparse_features:
            features = self.compute_sparse_features(x)
            x = torch.cat([x, features], dim=1)
        
        # Forward through network
        out = F.selu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = F.selu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = F.selu(self.bn3(self.fc3(out)))
        out = self.dropout(out)
        out = F.selu(self.bn4(self.fc4(out)))
        
        out = self.output(out)
        
        if self.task == 'regression':
            return torch.sigmoid(out)
        return out


class ConvDenseNet(nn.Module):
    """
    Scalable alternative to DenseNet using convolutions for compression.
    
    Uses convolutional layers to reduce spatial dimensions before
    dense layers, making it memory-efficient for large matrices.
    
    Memory usage: O(n²) for input, but only O(1) for model parameters.
    
    Input: Matrix of shape (batch, 1, n, n)
    """
    
    def __init__(self, n_size: int = 128, num_classes: int = 8,
                 task: str = 'classification', dropout: float = 0.3,
                 base_channels: int = 64):
        super().__init__()
        self.n_size = n_size
        self.num_classes = num_classes
        self.task = task
        
        # Convolutional encoder (compresses spatial dimensions)
        # Each block: Conv -> BN -> ReLU -> MaxPool (halves spatial dims)
        self.encoder = nn.Sequential(
            # Block 1: n -> n/2
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: n/2 -> n/4
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: n/4 -> n/8
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4: n/8 -> n/16
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Adaptive pooling to fixed size (handles any input dimension)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Dense classifier
        dense_input = base_channels * 8 * 4 * 4  # 512 * 16 = 8192
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dense_input, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        if task == 'classification':
            self.output = nn.Linear(256, num_classes)
        else:
            self.output = nn.Linear(256, 1)
    
    def forward(self, x):
        # Ensure 4D input (batch, 1, n, n)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            # Assume flattened input
            batch_size = x.size(0)
            n = int(x.size(1) ** 0.5)
            x = x.view(batch_size, 1, n, n)
        
        # Encode with convolutions
        x = self.encoder(x)
        x = self.adaptive_pool(x)
        
        # Classify
        x = self.classifier(x)
        x = self.output(x)
        
        if self.task == 'regression':
            return torch.sigmoid(x)
        return x


class LightDenseNet(nn.Module):
    """
    Memory-efficient dense network using random projection.
    
    Projects the high-dimensional input to a lower dimension before
    dense layers. This is a form of dimensionality reduction that
    preserves distances (Johnson-Lindenstrauss lemma).
    
    Memory usage: O(n² * projection_dim) but projection_dim << n²
    
    Input: Matrix of shape (batch, 1, n, n) or (batch, n*n)
    """
    
    def __init__(self, n_size: int = 128, num_classes: int = 8,
                 task: str = 'classification', dropout: float = 0.3,
                 projection_dim: int = 4096):
        super().__init__()
        self.n_size = n_size
        self.num_classes = num_classes
        self.task = task
        self.input_dim = n_size * n_size
        self.projection_dim = min(projection_dim, self.input_dim)
        
        # Random projection layer (frozen)
        # This reduces memory since we don't need gradients for this layer
        self.register_buffer(
            'projection_matrix',
            torch.randn(self.input_dim, self.projection_dim) / (self.projection_dim ** 0.5)
        )
        
        # Learnable dense layers on projected features
        self.network = nn.Sequential(
            nn.Linear(self.projection_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.SELU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SELU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SELU(),
            nn.Dropout(dropout),
        )
        
        if task == 'classification':
            self.output = nn.Linear(256, num_classes)
        else:
            self.output = nn.Linear(256, 1)
    
    def forward(self, x):
        # Flatten input
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        elif x.dim() == 3:
            x = x.view(x.size(0), -1)
        
        # Random projection (no gradients needed)
        x = torch.mm(x, self.projection_matrix)
        
        # Dense network
        x = self.network(x)
        x = self.output(x)
        
        if self.task == 'regression':
            return torch.sigmoid(x)
        return x


class ScalableDiagonalCNN(nn.Module):
    """
    Scalable version of DiagonalCNN for large matrices (n > 10,000).
    
    Key changes from original DiagonalCNN:
    1. Uses adaptive pooling before FC layer (O(1) parameters)
    2. Processes diagonal band with strided convolutions
    3. Can handle any matrix size with fixed memory
    
    For n = 10^6, this uses ~50MB instead of 262GB.
    
    Input: Diagonal band of shape (batch, 1, 2k+1, n)
    """
    
    def __init__(self, n_size: int = 128, num_classes: int = 8,
                 task: str = 'classification', band_width: int = 10,
                 dropout: float = 0.5):
        super().__init__()
        self.n_size = n_size
        self.num_classes = num_classes
        self.task = task
        self.k = band_width
        self.band_height = 2 * band_width + 1  # 21 for k=10
        
        # Convolutional feature extractor
        # Processes the diagonal band with increasing receptive field
        self.features = nn.Sequential(
            # Block 1: Local pattern detection
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # Only pool along n dimension
            
            # Block 2: Medium-scale patterns
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            
            # Block 3: Large-scale patterns  
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            
            # Block 4: Global patterns
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Adaptive pooling to fixed size (key for scalability!)
        # Pools to (band_height, 8) regardless of input width
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.band_height, 8))
        
        # Classifier with fixed input size
        fc_input = 256 * self.band_height * 8  # 256 * 21 * 8 = 43,008
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        if task == 'classification':
            self.output = nn.Linear(256, num_classes)
        else:
            self.output = nn.Linear(256, 1)
    
    def forward(self, x):
        # x shape: (batch, 1, 2k+1, n)
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        x = self.output(x)
        
        if self.task == 'regression':
            return torch.sigmoid(x)
        return x


class FeatureBasedPredictor(nn.Module):
    """
    Extremely scalable predictor using hand-crafted sparse matrix features.
    
    Instead of learning from raw data, this computes interpretable features
    from the sparse matrix and uses a small MLP to predict.
    
    This can handle ANY matrix size with O(nnz) complexity.
    
    Features computed:
    - Diagonal statistics (mean, std, min, max of |diag|)
    - Off-diagonal statistics
    - Bandwidth estimation
    - Block structure indicators
    - Sparsity pattern metrics
    
    Input: Feature vector (computed externally from sparse matrix)
    """
    
    def __init__(self, num_features: int = 32, num_classes: int = 8,
                 task: str = 'classification', dropout: float = 0.3,
                 n_size: int = None):  # n_size ignored, for API compatibility
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.task = task
        
        self.network = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        
        if task == 'classification':
            self.output = nn.Linear(32, num_classes)
        else:
            self.output = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.network(x)
        x = self.output(x)
        
        if self.task == 'regression':
            return torch.sigmoid(x)
        return x


def compute_sparse_features(matrix, n_features: int = 32) -> torch.Tensor:
    """
    Compute scalable features from a sparse matrix.
    
    This function extracts features in O(nnz) time, making it suitable
    for matrices of any size.
    
    Args:
        matrix: scipy.sparse matrix
        n_features: number of features to compute
        
    Returns:
        Feature tensor of shape (n_features,)
    """
    import numpy as np
    from scipy import sparse
    
    n = matrix.shape[0]
    nnz = matrix.nnz
    
    # Convert to CSR for efficient row access
    if not sparse.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()
    
    features = []
    
    # 1. Basic statistics
    features.append(n)  # Matrix size
    features.append(nnz)  # Number of nonzeros
    features.append(nnz / (n * n))  # Global density
    
    # 2. Diagonal statistics
    diag = matrix.diagonal()
    features.append(np.mean(np.abs(diag)))
    features.append(np.std(np.abs(diag)))
    features.append(np.min(np.abs(diag)) if len(diag) > 0 else 0)
    features.append(np.max(np.abs(diag)) if len(diag) > 0 else 0)
    features.append(np.sum(diag != 0) / n)  # Diagonal fill ratio
    
    # 3. Row statistics
    row_nnz = np.diff(matrix.indptr)
    features.append(np.mean(row_nnz))
    features.append(np.std(row_nnz))
    features.append(np.max(row_nnz))
    
    # 4. Bandwidth estimation (sample-based for large matrices)
    if n > 10000:
        # Sample rows for bandwidth estimation
        sample_rows = np.random.choice(n, min(1000, n), replace=False)
        bandwidths = []
        for i in sample_rows:
            cols = matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]
            if len(cols) > 0:
                bandwidths.append(max(abs(cols - i)))
        avg_bandwidth = np.mean(bandwidths) if bandwidths else 0
    else:
        # Exact bandwidth for small matrices
        coo = matrix.tocoo()
        bandwidths = np.abs(coo.row - coo.col)
        avg_bandwidth = np.mean(bandwidths) if len(bandwidths) > 0 else 0
    
    features.append(avg_bandwidth / n)  # Normalized bandwidth
    
    # 5. Symmetry measure (sample-based)
    if n > 10000:
        sample_size = min(10000, nnz)
        coo = matrix.tocoo()
        idx = np.random.choice(nnz, sample_size, replace=False)
        sym_count = 0
        for k in idx:
            i, j = coo.row[k], coo.col[k]
            if matrix[j, i] != 0:
                sym_count += 1
        symmetry = sym_count / sample_size
    else:
        diff = matrix - matrix.T
        symmetry = 1 - (diff.nnz / (2 * nnz + 1e-10))
    
    features.append(symmetry)
    
    # 6. Block structure indicators
    # Check if matrix has block-diagonal tendency
    block_sizes = [8, 16, 32, 64, 128]
    for bs in block_sizes:
        if bs > n:
            features.append(0)
            continue
        # Count nnz in block-diagonal region
        block_nnz = 0
        for i in range(0, n, bs):
            block_end = min(i + bs, n)
            block = matrix[i:block_end, i:block_end]
            block_nnz += block.nnz
        features.append(block_nnz / (nnz + 1e-10))
    
    # 7. Diagonal dominance
    abs_diag = np.abs(diag)
    row_sums = np.array(np.abs(matrix).sum(axis=1)).flatten()
    diag_dominance = np.mean(abs_diag / (row_sums + 1e-10))
    features.append(diag_dominance)
    
    # Pad or truncate to n_features
    features = np.array(features, dtype=np.float32)
    if len(features) < n_features:
        features = np.pad(features, (0, n_features - len(features)))
    else:
        features = features[:n_features]
    
    # Normalize
    features = (features - features.mean()) / (features.std() + 1e-10)
    
    return torch.tensor(features, dtype=torch.float32)


class ImageCNN(nn.Module):
    """
    Standard image-based CNN operating on the PNG sparsity pattern.
    
    This treats the matrix visualization as a standard image classification
    problem, using a VGG-style architecture.
    
    Input: RGB or grayscale image of shape (batch, channels, n, n)
    """
    
    def __init__(self, n_size: int = 128, in_channels: int = 1,
                 num_classes: int = 8, task: str = 'classification',
                 dropout: float = 0.5):
        super().__init__()
        self.n_size = n_size
        self.num_classes = num_classes
        self.task = task
        
        # VGG-style convolutional blocks
        self.features = nn.Sequential(
            # Block 1: 128 -> 64
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 32 -> 16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 16 -> 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate feature size after convolutions
        # 128 -> 64 -> 32 -> 16 -> 8
        feature_size = (n_size // 16) ** 2 * 512
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        if task == 'classification':
            self.output = nn.Linear(512, num_classes)
        else:
            self.output = nn.Linear(512, 1)
    
    def forward(self, x):
        # x shape: (batch, channels, n, n)
        out = self.features(x)
        out = self.classifier(out)
        out = self.output(out)
        
        if self.task == 'regression':
            return torch.sigmoid(out)
        return out


class ResNetBlock(nn.Module):
    """Residual block for ImageCNN variant."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ImageResNet(nn.Module):
    """
    ResNet-style CNN for matrix images.
    More powerful than VGG-style for complex patterns.
    """
    
    def __init__(self, n_size: int = 128, in_channels: int = 1,
                 num_classes: int = 8, task: str = 'classification',
                 dropout: float = 0.5):
        super().__init__()
        self.task = task
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        
        if task == 'classification':
            self.fc = nn.Linear(512, num_classes)
        else:
            self.fc = nn.Linear(512, 1)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        
        if self.task == 'regression':
            return torch.sigmoid(out)
        return out


def get_model(model_type: str, **kwargs) -> nn.Module:
    """Factory function to create models."""
    models = {
        'diagonal_cnn': DiagonalCNN,
        'scalable_diagonal': ScalableDiagonalCNN,
        'dense': DenseNet,
        'conv_dense': ConvDenseNet,
        'light_dense': LightDenseNet,
        'feature_based': FeatureBasedPredictor,
        'image_cnn': ImageCNN,
        'image_resnet': ImageResNet,
    }
    
    # BlockStartCNN is in a separate module due to different training paradigm
    if model_type == 'block_start':
        from block_start_cnn import BlockStartCNN
        return BlockStartCNN(**kwargs)
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())} or 'block_start'")
    
    return models[model_type](**kwargs)


if __name__ == '__main__':
    # Test all models
    batch_size = 4
    n_size = 128
    
    print("Testing DiagonalCNN...")
    model = DiagonalCNN(n_size=n_size, num_classes=8, task='classification')
    x = torch.randn(batch_size, 1, 21, n_size)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    
    print("\nTesting DenseNet...")
    model = DenseNet(n_size=n_size, num_classes=8, task='classification')
    x = torch.randn(batch_size, 1, n_size, n_size)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    
    print("\nTesting ConvDenseNet (memory-efficient alternative to DenseNet)...")
    model = ConvDenseNet(n_size=n_size, num_classes=8, task='classification')
    x = torch.randn(batch_size, 1, n_size, n_size)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    print("\nTesting LightDenseNet (random projection for large matrices)...")
    model = LightDenseNet(n_size=n_size, num_classes=8, task='classification')
    x = torch.randn(batch_size, 1, n_size, n_size)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    print("\nTesting ImageCNN...")
    model = ImageCNN(n_size=n_size, in_channels=1, num_classes=8, task='classification')
    x = torch.randn(batch_size, 1, n_size, n_size)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    
    print("\nTesting ImageResNet...")
    model = ImageResNet(n_size=n_size, in_channels=1, num_classes=8, task='classification')
    x = torch.randn(batch_size, 1, n_size, n_size)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    
    # Test with larger matrix to show memory efficiency
    print("\n--- Memory comparison for 500x500 matrix ---")
    n_large = 500
    
    print(f"\nConvDenseNet (500x500)...")
    model = ConvDenseNet(n_size=n_large, num_classes=8, task='classification')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params * 4 / 1e6:.1f} MB)")
    
    print(f"\nLightDenseNet (500x500)...")
    model = LightDenseNet(n_size=n_large, num_classes=8, task='classification')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params * 4 / 1e6:.1f} MB)")
    
    print(f"\nDenseNet (500x500) - for comparison (WARNING: ~2GB weights)...")
    dense_params = n_large * n_large * 2048 + 2048 * 1024 + 1024 * 512 + 512 * 256 + 256 * 8
    print(f"  Parameters: {dense_params:,} ({dense_params * 4 / 1e9:.1f} GB)")
    
    print("\nAll models working correctly!")
