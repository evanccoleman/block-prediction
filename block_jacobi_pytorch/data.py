"""
Data loading utilities for Block-Jacobi preconditioner prediction.

Handles:
- Loading sparse matrices from .npz files
- Loading PNG images
- Extracting diagonal bands (as in Götz & Anzt)
- Creating PyTorch datasets for all three model types
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
from scipy import sparse

import torch
from torch.utils.data import Dataset, DataLoader


# Threshold values used in the performance data
THRESHOLD_VALUES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
THRESHOLD_TO_IDX = {t: i for i, t in enumerate(THRESHOLD_VALUES)}
IDX_TO_THRESHOLD = {i: t for i, t in enumerate(THRESHOLD_VALUES)}

# Regression normalization constants (targets are in [0.05, 0.40])
REGRESSION_MIN = 0.05
REGRESSION_MAX = 0.40
REGRESSION_RANGE = REGRESSION_MAX - REGRESSION_MIN  # 0.35


def normalize_regression_target(value: float) -> float:
    """Normalize regression target from [0.05, 0.40] to [0, 1]."""
    return (value - REGRESSION_MIN) / REGRESSION_RANGE


def denormalize_regression_prediction(value: float) -> float:
    """Denormalize regression prediction from [0, 1] to [0.05, 0.40]."""
    return value * REGRESSION_RANGE + REGRESSION_MIN


def load_sparse_matrix(npz_path: Union[str, Path]) -> sparse.csr_matrix:
    """Load a sparse matrix from an npz file."""
    data = np.load(npz_path)
    matrix = sparse.csr_matrix(
        (data['data'], data['indices'], data['indptr']),
        shape=tuple(data['shape'])
    )
    return matrix


def load_matrix_metadata(json_path: Union[str, Path]) -> Dict:
    """Load matrix metadata from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_diagonal_band(matrix: np.ndarray, band_width: int = 10) -> np.ndarray:
    """
    Extract the diagonal band from a matrix.
    
    Following Götz & Anzt (2018):
    - Extract elements within distance w from the main diagonal
    - Arrange row-wise, right-aligned
    - Fill missing values (left side) with zeros
    
    Args:
        matrix: Square matrix of shape (n, n)
        band_width: Half-width of the band (w). Total height = 2w + 1
    
    Returns:
        Diagonal image of shape (2*band_width + 1, n)
    """
    n = matrix.shape[0]
    diag_height = 2 * band_width + 1
    diagonal_image = np.zeros((diag_height, n), dtype=np.float32)
    
    for i in range(n):
        for offset in range(-band_width, band_width + 1):
            j = i + offset
            if 0 <= j < n:
                # Map to diagonal image coordinates
                # offset=-w maps to row 0, offset=0 maps to row w, offset=w maps to row 2w
                row = offset + band_width
                diagonal_image[row, i] = matrix[i, j]
    
    return diagonal_image


def normalize_matrix(matrix: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Normalize matrix values."""
    if method == 'minmax':
        min_val = matrix.min()
        max_val = matrix.max()
        if max_val - min_val > 1e-8:
            return (matrix - min_val) / (max_val - min_val)
        return matrix - min_val
    elif method == 'zscore':
        mean = matrix.mean()
        std = matrix.std()
        if std > 1e-8:
            return (matrix - mean) / std
        return matrix - mean
    elif method == 'binary':
        return (matrix != 0).astype(np.float32)
    else:
        return matrix


class BlockJacobiDataset(Dataset):
    """
    Dataset for Block-Jacobi preconditioner parameter prediction.
    
    Supports multiple input types:
    - 'diagonal': Diagonal band extraction (for DiagonalCNN)
    - 'matrix': Full matrix (for DenseNet)
    - 'image': PNG image (for ImageCNN)
    
    Supports two directory structures:
    1. Flat structure (all files in one directory):
       data_dir/matrix_0.npz, matrix_0.json, matrix_0.png
       
    2. Subfolder structure (your png_builder2.py format):
       data_dir/
         images/matrix_0.png
         matrices/matrix_0.npz
         metadata/matrix_0.json
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        input_type: str = 'diagonal',
        task: str = 'classification',
        target_metric: str = 'iterations',  # 'iterations' or 'time'
        band_width: int = 10,
        normalize: str = 'binary',
        transform=None,
    ):
        """
        Args:
            data_dir: Directory containing matrix data (flat or subfolder structure)
            input_type: One of 'diagonal', 'matrix', 'image'
            task: 'classification' or 'regression'
            target_metric: Which metric to optimize for ('iterations' or 'time')
            band_width: Band width for diagonal extraction
            normalize: Normalization method ('minmax', 'zscore', 'binary', None)
            transform: Optional torchvision transforms for image input
        """
        self.data_dir = Path(data_dir)
        self.input_type = input_type
        self.task = task
        self.target_metric = target_metric
        self.band_width = band_width
        self.normalize = normalize
        self.transform = transform
        
        # Detect directory structure and find all samples
        self.samples = self._find_samples()
        
    def _find_samples(self) -> List[Dict]:
        """Find all matrix samples in the data directory."""
        samples = []
        
        # Check for subfolder structure (images/, matrices/, metadata/)
        images_dir = self.data_dir / 'images'
        matrices_dir = self.data_dir / 'matrices'
        metadata_dir = self.data_dir / 'metadata'
        
        if metadata_dir.exists():
            # Subfolder structure (png_builder2.py format)
            json_files = sorted(metadata_dir.glob('matrix_*.json'))
            
            for json_path in json_files:
                stem = json_path.stem  # e.g., 'matrix_0'
                matrix_id = stem.split('_')[1]
                
                # Look for corresponding files
                png_path = images_dir / f'{stem}.png'
                npz_path = matrices_dir / f'{stem}.npz'
                
                sample = {
                    'id': matrix_id,
                    'json': json_path,
                    'png': png_path if png_path.exists() else None,
                    'npz': npz_path if npz_path.exists() else None,
                }
                
                # Must have either PNG or NPZ for input
                if sample['png'] or sample['npz']:
                    samples.append(sample)
                    
        else:
            # Flat structure (all files in same directory)
            # Try to find by JSON files first
            json_files = sorted(self.data_dir.glob('matrix_*.json'))
            
            if json_files:
                for json_path in json_files:
                    stem = json_path.stem
                    matrix_id = stem.split('_')[1]
                    
                    png_path = self.data_dir / f'{stem}.png'
                    npz_path = self.data_dir / f'{stem}.npz'
                    
                    sample = {
                        'id': matrix_id,
                        'json': json_path,
                        'png': png_path if png_path.exists() else None,
                        'npz': npz_path if npz_path.exists() else None,
                    }
                    
                    if sample['png'] or sample['npz']:
                        samples.append(sample)
            else:
                # Fallback: find by NPZ files
                npz_files = sorted(self.data_dir.glob('matrix_*.npz'))
                
                for npz_path in npz_files:
                    stem = npz_path.stem
                    matrix_id = stem.split('_')[1]
                    
                    json_path = self.data_dir / f'{stem}.json'
                    png_path = self.data_dir / f'{stem}.png'
                    
                    if json_path.exists():
                        samples.append({
                            'id': matrix_id,
                            'npz': npz_path,
                            'json': json_path,
                            'png': png_path if png_path.exists() else None
                        })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load metadata for labels
        metadata = load_matrix_metadata(sample['json'])
        
        # Get label based on task
        if self.task == 'classification':
            if self.target_metric == 'iterations':
                optimal_threshold = metadata['labels']['class_optimal_iterations']
            else:
                optimal_threshold = metadata['labels']['class_optimal_time']
            label = torch.tensor(THRESHOLD_TO_IDX[optimal_threshold], dtype=torch.long)
        else:  # regression
            # Use interpolated optimal for regression (more continuous)
            if 'regression_interpolated_optimal' in metadata['labels']:
                label_val = metadata['labels']['regression_interpolated_optimal']
            else:
                label_val = metadata['labels'].get('regression_ground_truth', 
                                                    metadata['labels']['class_optimal_iterations'])
            # Normalize to [0, 1] for sigmoid output
            label_val_normalized = normalize_regression_target(label_val)
            label = torch.tensor([label_val_normalized], dtype=torch.float32)
        
        # Load input based on type and available files
        if self.input_type == 'image':
            # Prefer PNG for image input
            if sample.get('png') and sample['png'].exists():
                img = Image.open(sample['png']).convert('L')  # Grayscale
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                if self.transform:
                    img = self.transform(img)
                    data = img
                else:
                    # Invert so nonzeros are 1, zeros are 0
                    img_array = 1.0 - img_array
                    data = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)
            elif sample.get('npz') and sample['npz'].exists():
                # Fallback: create binary image from matrix
                matrix = load_sparse_matrix(sample['npz'])
                dense = matrix.toarray().astype(np.float32)
                binary = (dense != 0).astype(np.float32)
                data = torch.tensor(binary, dtype=torch.float32).unsqueeze(0)
            else:
                raise ValueError(f"No image or matrix file found for sample {sample['id']}")
        else:
            # Need raw matrix for diagonal or matrix input
            if sample.get('npz') is None or not sample['npz'].exists():
                raise ValueError(f"Matrix input requires .npz file for sample {sample['id']}. "
                                f"Use --save_raw flag when generating data, or use input_type='image'.")
            
            matrix = load_sparse_matrix(sample['npz'])
            dense = matrix.toarray().astype(np.float32)
            
            # Normalize
            if self.normalize:
                dense = normalize_matrix(dense, self.normalize)
            
            if self.input_type == 'diagonal':
                # Extract diagonal band
                diagonal = extract_diagonal_band(dense, self.band_width)
                data = torch.tensor(diagonal, dtype=torch.float32).unsqueeze(0)  # (1, 21, n)
            else:  # 'matrix'
                data = torch.tensor(dense, dtype=torch.float32).unsqueeze(0)  # (1, n, n)
        
        return data, label


class SyntheticBlockJacobiDataset(Dataset):
    """
    Generate synthetic data similar to png_builder2.py.
    
    Generates matrices with:
    - Jittered block structure (target fraction ±20%)
    - High density within blocks (0.8)
    - Low off-diagonal noise
    - Controlled diagonal for solver convergence
    
    Note: This generates ONLY the matrices and assigns synthetic labels.
    For accurate labels based on actual solver performance, use png_builder2.py.
    """
    
    def __init__(
        self,
        num_samples: int = 3000,
        n_size: int = 128,
        input_type: str = 'diagonal',
        task: str = 'classification',
        band_width: int = 10,
        seed: Optional[int] = None,
        jitter_fraction: float = 0.2,
        block_density: float = 0.8,
        noise_range: Tuple[float, float] = (0.001, 0.02),
    ):
        self.num_samples = num_samples
        self.n_size = n_size
        self.input_type = input_type
        self.task = task
        self.band_width = band_width
        self.jitter_fraction = jitter_fraction
        self.block_density = block_density
        self.noise_range = noise_range
        
        if seed is not None:
            np.random.seed(seed)
        
        # Pre-generate all samples
        self.data, self.labels, self.target_fractions = self._generate_all()
    
    def _generate_jittered_matrix(self, target_block_fraction: float, noise_level: float) -> np.ndarray:
        """
        Generate a matrix with jittered block structure (matching png_builder2.py).
        """
        n = self.n_size
        rng = np.random.default_rng()
        
        # Calculate base block size and jitter range
        base_block_size = int(n * target_block_fraction)
        jitter = int(base_block_size * self.jitter_fraction)
        
        # Generate blocks with jittered sizes
        blocks = []
        current_dim = 0
        
        while current_dim < n:
            # Jitter the block size
            if jitter > 0:
                this_block_size = base_block_size + rng.integers(-jitter, jitter + 1)
            else:
                this_block_size = base_block_size
            this_block_size = max(2, this_block_size)
            
            # Clamp to remaining space
            if current_dim + this_block_size > n:
                this_block_size = n - current_dim
            
            # Create dense block with specified density
            block_data = rng.uniform(-10.0, 10.0, (this_block_size, this_block_size))
            block_mask = rng.random((this_block_size, this_block_size)) < self.block_density
            block = np.where(block_mask, block_data, 0.0)
            blocks.append(block)
            
            current_dim += this_block_size
        
        # Assemble block diagonal matrix
        matrix = np.zeros((n, n), dtype=np.float32)
        current_pos = 0
        for block in blocks:
            size = block.shape[0]
            matrix[current_pos:current_pos+size, current_pos:current_pos+size] = block
            current_pos += size
        
        # Add off-diagonal noise
        if noise_level > 0:
            noise_mask = rng.random((n, n)) < noise_level
            noise_values = rng.uniform(-0.5, 0.5, (n, n))
            matrix = matrix + np.where(noise_mask, noise_values, 0.0)
        
        # Set diagonal for numerical stability (weak but non-singular)
        diag_values = rng.uniform(15.0, 25.0, n)
        off_diag_sum = np.abs(matrix).sum(axis=1) - np.abs(np.diag(matrix))
        diag_values = diag_values + 0.1 * off_diag_sum
        diag_signs = rng.choice([-1, 1], size=n)
        np.fill_diagonal(matrix, diag_values * diag_signs)
        
        return matrix.astype(np.float32)
    
    def _generate_all(self) -> Tuple[List[np.ndarray], List, List[float]]:
        """Generate all samples."""
        data = []
        labels = []
        target_fractions = []
        
        for _ in range(self.num_samples):
            # Random target structure fraction (uniform in [0.05, 0.40])
            target_fraction = np.random.uniform(0.05, 0.40)
            noise_level = np.random.uniform(*self.noise_range)
            
            matrix = self._generate_jittered_matrix(target_fraction, noise_level)
            
            # Process for input type
            if self.input_type == 'diagonal':
                processed = extract_diagonal_band(matrix, self.band_width)
            else:
                processed = matrix
            
            data.append(processed)
            target_fractions.append(target_fraction)
            
            # Generate synthetic label
            # Note: Real labels should come from actual solver profiling!
            if self.task == 'classification':
                # Map target fraction to nearest class
                nearest_class_idx = np.argmin([abs(target_fraction - t) for t in THRESHOLD_VALUES])
                labels.append(nearest_class_idx)
            else:
                # Regression: normalize target fraction to [0, 1]
                normalized = normalize_regression_target(target_fraction)
                labels.append(np.array([normalized], dtype=np.float32))
        
        return data, labels, target_fractions
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        
        # Add channel dimension
        data = data.unsqueeze(0)
        
        if self.task == 'classification':
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return data, label


def create_dataloaders(
    data_dir: Union[str, Path],
    input_type: str = 'diagonal',
    task: str = 'classification',
    batch_size: int = 8,
    val_split: float = 0.2,
    num_workers: int = 0,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Path to data directory
        input_type: 'diagonal', 'matrix', or 'image'
        task: 'classification' or 'regression'
        batch_size: Batch size
        val_split: Fraction of data for validation
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for dataset
    
    Returns:
        train_loader, val_loader
    """
    dataset = BlockJacobiDataset(
        data_dir=data_dir,
        input_type=input_type,
        task=task,
        **dataset_kwargs
    )
    
    # Split into train/val
    n_samples = len(dataset)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test with the provided sample data
    print("Testing data loading...")
    
    # Test loading single matrix
    npz_path = '/mnt/user-data/uploads/matrix_0.npz'
    json_path = '/mnt/user-data/uploads/matrix_0.json'
    
    matrix = load_sparse_matrix(npz_path)
    metadata = load_matrix_metadata(json_path)
    
    print(f"Matrix shape: {matrix.shape}, nnz: {matrix.nnz}")
    print(f"Labels: {metadata['labels']}")
    
    # Test diagonal extraction
    dense = matrix.toarray()
    diagonal = extract_diagonal_band(dense, band_width=10)
    print(f"Diagonal band shape: {diagonal.shape}")
    
    # Test synthetic dataset
    print("\nTesting synthetic dataset...")
    dataset = SyntheticBlockJacobiDataset(
        num_samples=10,
        input_type='diagonal',
        task='classification',
        seed=42
    )
    
    data, label = dataset[0]
    print(f"Synthetic sample - data shape: {data.shape}, label shape: {label.shape}")
