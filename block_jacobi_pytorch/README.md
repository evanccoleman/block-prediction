# Block-Jacobi Preconditioner Prediction with PyTorch

This is a PyTorch reimplementation of the CNN-based block pattern detection approach from:

> Götz & Anzt (2018). "Machine Learning-Aided Numerical Linear Algebra: Convolutional Neural Networks for the Efficient Preconditioner Generation"

## Overview

This package implements multiple model architectures for predicting optimal block-Jacobi preconditioner parameters:

### Global Prediction Models (predict one block size for entire matrix)

| Model | Description | Params | Scalability |
|-------|-------------|--------|-------------|
| **DiagonalCNN** | Original paper architecture - extracts diagonal band | O(n) | n < 10,000 |
| **ScalableDiagonalCNN** | Adaptive pooling version | O(1) | Any n |
| **ConvDenseNet** | Conv compression + dense layers | O(1) | Any n |
| **ImageResNet** | ResNet on matrix spy image | O(1) | Any n |

### Per-Entry Prediction (original paper formulation)

| Model | Description | Params | Scalability |
|-------|-------------|--------|-------------|
| **BlockStartCNN** | Predicts block start at each diagonal position | O(1) | Any n (O(n) inference) |

The **BlockStartCNN** is closest to the original paper's formulation and supports **variable block sizes**.

## Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Train a single model
```bash
# Classification (8-class: which block fraction is optimal?)
python train.py --model diagonal_cnn --task classification --data-dir ./your_data/

# Regression (predict continuous optimal threshold)
python train.py --model scalable_diagonal --task regression --data-dir ./your_data/
```

### Run full comparison of all models
```bash
./scripts/run_full_comparison.sh ./your_data/ 30  # 30 epochs
```

### Train BlockStartCNN (per-entry prediction)
```bash
python block_start_cnn.py --data-dir ./your_data/ --epochs 50
```

## Model Selection Guide

| Your Situation | Recommended Model |
|----------------|-------------------|
| Small matrices (n < 1,000), best accuracy | `diagonal_cnn` |
| Medium matrices (1,000 < n < 10,000) | `scalable_diagonal` |
| Large matrices (n > 10,000) | `scalable_diagonal` with tiling |
| Variable block sizes needed | `block_start_cnn` |
| Image-based workflow | `image_resnet` |
| Limited GPU memory | `scalable_diagonal` or `conv_dense` |

## Data Format

### Directory Structure
```
data_dir/
├── matrices/           # Optional subfolder
│   ├── matrix_0.npz   
│   └── ...
├── metadata/           # Optional subfolder
│   ├── matrix_0.json
│   └── ...
├── images/             # Optional subfolder  
│   ├── matrix_0.png
│   └── ...
```

Or flat structure:
```
data_dir/
├── matrix_0.npz
├── matrix_0.json
├── matrix_0.png
└── ...
```

### Matrix Files (.npz)
Sparse matrices in SciPy CSR format:
```python
import scipy.sparse as sp
matrix = sp.load_npz('matrix_0.npz')
```

### Metadata Files (.json)
```json
{
  "matrix_id": 0,
  "labels": {
    "class_optimal_time": 0.35,
    "regression_ground_truth": 0.274
  },
  "matrix_properties": {
    "size": 128,
    "nnz": 3368
  }
}
```

## Scalability Analysis

The key bottleneck in the original DiagonalCNN is the FC layer after flattening:

```
DiagonalCNN FC layer: 128 * n → 512 parameters
  n = 128:     8 million params (OK)
  n = 10,000:  655 million params (problematic)
  n = 1,000,000: 65 billion params (impossible)
```

**Solutions implemented:**

1. **ScalableDiagonalCNN**: Uses adaptive pooling before FC layer → O(1) params
2. **Tiling**: Process 128×128 tiles, aggregate predictions
3. **BlockStartCNN**: Sliding window, same model for any position

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/run_full_comparison.sh` | Compare all models (classification + regression) |
| `scripts/run_classification_vs_regression.sh` | Compare tasks |
| `scripts/run_improved.sh` | Training with heavy regularization |
| `analyze_resolution.py` | Analyze downsampling effects |
| `block_start_cnn.py` | Per-entry prediction training |

## Citation

```bibtex
@inproceedings{gotz2018machine,
  title={Machine Learning-Aided Numerical Linear Algebra: Convolutional Neural Networks for the Efficient Preconditioner Generation},
  author={G{\"o}tz, Markus and Anzt, Hartwig},
  booktitle={SC18: International Conference for High Performance Computing, Networking, Storage and Analysis},
  year={2018}
}
```
