# Block-Jacobi Preconditioner Prediction with PyTorch

This is a PyTorch reimplementation of the CNN-based block pattern detection approach from:

> Götz & Anzt (2018). "Machine Learning-Aided Numerical Linear Algebra: Convolutional Neural Networks for the Efficient Preconditioner Generation"

## Overview

This package implements three model architectures for predicting optimal block-Jacobi preconditioner parameters:

1. **DiagonalCNN** - The original approach from the paper: extracts the diagonal band from the matrix and uses a specialized CNN architecture with residual denoising and corner detection convolutions.

2. **DenseNet** - A dense neural network that operates on the full raw matrix data, treating it as a high-dimensional feature vector.

3. **ImageCNN/ImageResNet** - Standard image classification CNNs (VGG-style and ResNet-style) that operate on the PNG visualization of the matrix sparsity pattern.

## Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- SciPy
- Pillow
- matplotlib (for experiments/visualization)

## Usage

### Training with Synthetic Data

The package can generate synthetic matrices with block structure similar to the paper:

```bash
# Train DiagonalCNN (paper's approach)
python train.py --model diagonal_cnn --task classification --epochs 50 --synthetic

# Train DenseNet on raw matrix data
python train.py --model dense --task classification --epochs 50 --synthetic

# Train ImageResNet on matrix images
python train.py --model image_resnet --task classification --epochs 50 --synthetic
```

### Training with Your Data

Organize your data as:
```
data_dir/
  matrix_0.npz    # Sparse matrix (scipy.sparse CSR format)
  matrix_0.json   # Metadata with labels
  matrix_0.png    # Matrix sparsity pattern image (optional)
  matrix_1.npz
  matrix_1.json
  ...
```

Then run:
```bash
python train.py --model diagonal_cnn --data-dir ./data_dir --epochs 50
```

### Running Predictions

```bash
python predict.py --model-path output/run_name/best_model.pt --input matrix_0.npz
```

### Comparing All Models

Run a full comparison experiment:
```bash
python experiments.py
```

This will train all three architectures on synthetic data and produce comparison plots.

## Data Format

### Matrix Files (.npz)
Sparse matrices stored in SciPy CSR format with keys:
- `data`: Non-zero values
- `indices`: Column indices
- `indptr`: Row pointer
- `shape`: Matrix dimensions
- `format`: 'csr'

### Metadata Files (.json)
```json
{
  "matrix_id": 0,
  "labels": {
    "class_optimal_time": 0.35,
    "class_optimal_iterations": 0.35,
    "regression_ground_truth": 0.274,
    "regression_interpolated_optimal": 0.247
  },
  "matrix_properties": {
    "size": 128,
    "nnz": 3368,
    "density": 0.206
  },
  "performance_data": {
    "0.05": {"iterations": 59, "total_wall": 0.024, ...},
    "0.10": {"iterations": 51, "total_wall": 0.020, ...},
    ...
  }
}
```

## Model Architectures

### DiagonalCNN (Paper Approach)
```
Input: Diagonal band (21 × 128)
  → Residual Denoising Block (BatchNorm + SELU + Conv2D)
  → Corner Detection (padded convolution + tanh)
  → Dense Classifier (512 → output)
```

### DenseNet
```
Input: Full matrix (128 × 128)
  → Optional sparse features (nnz, density, diag_dominance)
  → FC layers: 2048 → 1024 → 512 → 256 → output
  → SELU activations, BatchNorm, Dropout
```

### ImageCNN / ImageResNet
```
Input: Matrix image (128 × 128)
  → VGG-style or ResNet-style conv blocks
  → Global pooling
  → Dense classifier
```

## Task Types

- **Classification**: Predict the optimal threshold from discrete options (0.05, 0.10, ..., 0.40)
- **Regression**: Predict the continuous optimal threshold value
- **Block Start Prediction**: Multi-label classification to predict where blocks start (original paper task)

## Key Differences from Original

1. **Framework**: PyTorch instead of TensorFlow/Keras
2. **Additional Models**: Added DenseNet and ImageCNN/ImageResNet variants
3. **Flexible Tasks**: Support for both classification and regression
4. **Modern Optimizers**: Uses NAdam (Nesterov Adam) as in the paper

## File Structure

```
block_jacobi_pytorch/
├── models.py          # Model architectures
├── data.py            # Data loading and preprocessing
├── train.py           # Training script
├── predict.py         # Inference script
├── experiments.py     # Model comparison experiments
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{gotz2018machine,
  title={Machine Learning-Aided Numerical Linear Algebra: Convolutional Neural Networks for the Efficient Preconditioner Generation},
  author={G{\"o}tz, Markus and Anzt, Hartwig},
  booktitle={Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  year={2018}
}
```

## License

This is a research reimplementation for educational purposes.
