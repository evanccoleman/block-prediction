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
        'dense': DenseNet,
        'image_cnn': ImageCNN,
        'image_resnet': ImageResNet,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
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
    
    print("\nAll models working correctly!")
