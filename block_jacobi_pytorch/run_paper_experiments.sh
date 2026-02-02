#!/bin/bash
# run_paper_experiments.sh - Run ALL experiments needed for the paper
#
# Usage: ./scripts/run_paper_experiments.sh
#
# Prerequisites:
#   - Datasets generated with png_builder2.py:
#     - dataset_128_gmres/   (small, GMRES-labeled)
#     - dataset_500_gmres/   (medium, GMRES-labeled) 
#     - dataset_2000_theo/   (large, theoretical)
#     - dataset_10000_theo/  (XL, theoretical)
#
# This script runs experiments to populate:
#   - Table 2: Classification accuracy
#   - Table 3: Regression MAE
#   - Table 4: Transfer across matrix sizes
#   - Figure 5: Scalability analysis

set -e

# ============================================
# CONFIGURATION
# ============================================

# Dataset paths (adjust these to your setup)
DATASET_128="${DATASET_128:-./dataset_128_gmres}"
DATASET_500="${DATASET_500:-./dataset_500_gmres}"
DATASET_2000="${DATASET_2000:-./dataset_2000_theo}"
DATASET_10000="${DATASET_10000:-./dataset_10000_theo}"

EPOCHS=${EPOCHS:-30}
OUTPUT_BASE="./experiments/paper_results"

mkdir -p "$OUTPUT_BASE"

echo "============================================================"
echo "BLOCK-JACOBI CNN PAPER EXPERIMENTS"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  DATASET_128:  $DATASET_128"
echo "  DATASET_500:  $DATASET_500"
echo "  DATASET_2000: $DATASET_2000"
echo "  DATASET_10000: $DATASET_10000"
echo "  EPOCHS: $EPOCHS"
echo "  OUTPUT: $OUTPUT_BASE"
echo ""

# Check datasets exist
MISSING=0
for ds in "$DATASET_128" "$DATASET_500" "$DATASET_2000" "$DATASET_10000"; do
    if [ ! -d "$ds" ]; then
        echo "WARNING: Dataset not found: $ds"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "Some datasets are missing. You can:"
    echo "  1. Generate them with png_builder2.py"
    echo "  2. Set environment variables: DATASET_128=/path/to/data ..."
    echo "  3. Continue anyway (experiments on missing data will be skipped)"
    echo ""
    read -p "Continue? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ============================================
# EXPERIMENT 1: Full Model Comparison (Table 2 & 3)
# ============================================
echo ""
echo "============================================================"
echo "EXPERIMENT 1: Full Model Comparison"
echo "  → Populates Table 2 (Classification) and Table 3 (Regression)"
echo "============================================================"

# Run on primary dataset (128 with GMRES labels)
if [ -d "$DATASET_128" ]; then
    echo ""
    echo "Running on n=128 dataset..."
    OUTPUT_DIR="$OUTPUT_BASE/comparison_n128"
    
    if [ -f "$OUTPUT_DIR/comparison_results.csv" ]; then
        echo "  Already complete, skipping (delete $OUTPUT_DIR to re-run)"
    else
        ./scripts/run_full_comparison.sh "$DATASET_128" $EPOCHS
        mv ./experiments/full_comparison "$OUTPUT_DIR"
    fi
fi

# Optionally run on n=500 for validation
if [ -d "$DATASET_500" ]; then
    echo ""
    echo "Running on n=500 dataset..."
    OUTPUT_DIR="$OUTPUT_BASE/comparison_n500"
    
    if [ -f "$OUTPUT_DIR/comparison_results.csv" ]; then
        echo "  Already complete, skipping"
    else
        ./scripts/run_full_comparison.sh "$DATASET_500" $EPOCHS
        mv ./experiments/full_comparison "$OUTPUT_DIR"
    fi
fi

# ============================================
# EXPERIMENT 2: Transfer Learning (Table 4)
# ============================================
echo ""
echo "============================================================"
echo "EXPERIMENT 2: Transfer Learning Study"
echo "  → Populates Table 4 (Transfer across matrix sizes)"
echo "============================================================"

# Train on n=128, test on all sizes
TRANSFER_ARGS=""
[ -d "$DATASET_128" ] && TRANSFER_ARGS="$DATASET_128"
[ -d "$DATASET_500" ] && TRANSFER_ARGS="$TRANSFER_ARGS $DATASET_500"
[ -d "$DATASET_2000" ] && TRANSFER_ARGS="$TRANSFER_ARGS $DATASET_2000"
[ -d "$DATASET_10000" ] && TRANSFER_ARGS="$TRANSFER_ARGS $DATASET_10000"

if [ -n "$TRANSFER_ARGS" ]; then
    if [ -f "$OUTPUT_BASE/transfer_study/transfer_results.csv" ]; then
        echo "  Already complete, skipping"
    else
        ./scripts/run_transfer_study.sh $TRANSFER_ARGS
        mv ./experiments/transfer_study "$OUTPUT_BASE/transfer_study"
    fi
else
    echo "  Skipping: no datasets available"
fi

# ============================================
# EXPERIMENT 3: Scalability Analysis (Figure 5)
# ============================================
echo ""
echo "============================================================"
echo "EXPERIMENT 3: Parameter Scalability Analysis"
echo "  → Populates Figure 5 (Parameter counts vs n)"
echo "============================================================"

# This is analytical - compute parameter counts for different n
python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path
import sys
sys.path.insert(0, '.')

try:
    from models import DiagonalCNN, ScalableDiagonalCNN, ConvDenseNet, ImageResNet
except ImportError:
    print("  Could not import models, skipping parameter analysis")
    sys.exit(0)

output_dir = Path("./experiments/paper_results/scalability")
output_dir.mkdir(parents=True, exist_ok=True)

# Matrix sizes to analyze
sizes = [128, 256, 500, 1000, 2000, 5000, 10000]

results = {
    'sizes': sizes,
    'models': {}
}

print("\nParameter counts by matrix size:")
print("=" * 80)
print(f"{'Model':<25} " + " ".join(f"{n:>10}" for n in sizes))
print("-" * 80)

for model_class, name in [
    (DiagonalCNN, 'diagonal_cnn'),
    (ScalableDiagonalCNN, 'scalable_diagonal'),
    (ConvDenseNet, 'conv_dense'),
    (ImageResNet, 'image_resnet'),
]:
    params = []
    for n in sizes:
        try:
            if name == 'image_resnet':
                model = model_class(n_classes=8, image_size=128)
            else:
                model = model_class(matrix_size=n, n_classes=8)
            p = sum(p.numel() for p in model.parameters())
            params.append(p)
        except Exception as e:
            params.append(None)
    
    results['models'][name] = params
    
    # Print row
    param_strs = []
    for p in params:
        if p is None:
            param_strs.append("OOM")
        elif p > 1e9:
            param_strs.append(f"{p/1e9:.1f}B")
        elif p > 1e6:
            param_strs.append(f"{p/1e6:.1f}M")
        elif p > 1e3:
            param_strs.append(f"{p/1e3:.0f}K")
        else:
            param_strs.append(str(p))
    
    print(f"{name:<25} " + " ".join(f"{s:>10}" for s in param_strs))

print("-" * 80)
print("\nKey observations:")
print("  - diagonal_cnn: O(n) parameters (FC layer bottleneck)")
print("  - scalable_diagonal: O(1) parameters (adaptive pooling)")
print("  - conv_dense: O(1) parameters (conv compression)")
print("  - image_resnet: O(1) parameters (fixed image size)")

# Save results
with open(output_dir / 'parameter_scaling.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_dir / 'parameter_scaling.json'}")
PYTHON_SCRIPT

# ============================================
# EXPERIMENT 4: Sample Efficiency (Optional)
# ============================================
echo ""
echo "============================================================"
echo "EXPERIMENT 4: Sample Efficiency (Optional)"
echo "  → Shows accuracy vs training set size"
echo "============================================================"

if [ "${RUN_SAMPLE_EFFICIENCY:-0}" == "1" ]; then
    ./scripts/run_scalability_study.sh diagonal_cnn
    mv ./experiments/scalability "$OUTPUT_BASE/sample_efficiency"
else
    echo "  Skipping (set RUN_SAMPLE_EFFICIENCY=1 to enable)"
fi

# ============================================
# EXPERIMENT 5: Hyperparameter Sensitivity (Optional)
# ============================================
echo ""
echo "============================================================"
echo "EXPERIMENT 5: Hyperparameter Sensitivity (Optional)"
echo "  → Validates robustness to hyperparameter choices"
echo "============================================================"

if [ "${RUN_HYPERPARAM:-0}" == "1" ]; then
    ./scripts/run_hyperparam_sweep.sh diagonal_cnn
    mv ./experiments/hyperparam_sweep "$OUTPUT_BASE/hyperparam_sweep"
else
    echo "  Skipping (set RUN_HYPERPARAM=1 to enable)"
fi

# ============================================
# GENERATE FINAL SUMMARY
# ============================================
echo ""
echo "============================================================"
echo "GENERATING FINAL SUMMARY"
echo "============================================================"

python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path
import pandas as pd

output_base = Path("./experiments/paper_results")

print("\n" + "=" * 80)
print("PAPER RESULTS SUMMARY")
print("=" * 80)

# Table 2 & 3: Model comparison
comparison_dir = output_base / "comparison_n128"
if (comparison_dir / "comparison_results.csv").exists():
    print("\n--- TABLE 2 & 3: Model Comparison (n=128) ---")
    df = pd.read_csv(comparison_dir / "comparison_results.csv")
    
    print("\nClassification:")
    class_df = df[df['task'] == 'classification'][['model', 'val_metric', 'params']]
    class_df.columns = ['Model', 'Accuracy', 'Parameters']
    print(class_df.to_string(index=False))
    
    print("\nRegression:")
    reg_df = df[df['task'] == 'regression'][['model', 'val_metric']]
    reg_df.columns = ['Model', 'MAE']
    print(reg_df.to_string(index=False))

# Table 4: Transfer
transfer_file = output_base / "transfer_study" / "transfer_results.csv"
if transfer_file.exists():
    print("\n--- TABLE 4: Transfer Learning ---")
    df = pd.read_csv(transfer_file)
    pivot = df.pivot_table(
        index='model',
        columns='test_size', 
        values='test_accuracy',
        aggfunc='first'
    )
    print(pivot.to_string())

# Figure 5: Scalability
scalability_file = output_base / "scalability" / "parameter_scaling.json"
if scalability_file.exists():
    print("\n--- FIGURE 5: Parameter Scaling ---")
    with open(scalability_file) as f:
        data = json.load(f)
    print("See parameter_scaling.json for full data")
    print("Key: diagonal_cnn is O(n), others are O(1)")

print("\n" + "=" * 80)
print(f"Full results in: {output_base}")
print("=" * 80)
PYTHON_SCRIPT

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Results are in: $OUTPUT_BASE/"
echo ""
echo "To cite in paper:"
echo "  - Table 2: comparison_n128/comparison_results.csv (classification)"
echo "  - Table 3: comparison_n128/comparison_results.csv (regression)"
echo "  - Table 4: transfer_study/transfer_results.csv"
echo "  - Figure 5: scalability/parameter_scaling.json"
echo ""
