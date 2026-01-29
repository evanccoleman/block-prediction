#!/bin/bash
# run_improved.sh - Train models with improved regularization
#
# Usage: ./scripts/run_improved.sh /path/to/data/directory
#
# Key improvements over basic training:
# - Early stopping (patience=15)
# - Label smoothing (0.1)
# - Higher weight decay (0.05)
# - Mixup augmentation (alpha=0.2)
# - Higher dropout (0.3-0.5)
# - Gradient clipping

set -e

DATA_DIR=${1:?"Usage: $0 /path/to/data/directory"}
OUTPUT_DIR="./experiments/improved"
EPOCHS=100
BATCH_SIZE=16
LR=1e-4

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Improved Training with Regularization"
echo "=============================================="
echo "Data directory: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# ============================================
# DiagonalCNN - Already performs well, light regularization
# ============================================
echo "Training DiagonalCNN (light regularization)..."
python train_improved.py \
    --model diagonal_cnn \
    --data-dir "$DATA_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --weight-decay 0.02 \
    --dropout 0.2 \
    --label-smoothing 0.05 \
    --mixup 0.1 \
    --patience 20 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/diagonal_cnn.log"

# ============================================
# DenseNet - Heavy regularization to prevent overfitting
# ============================================
echo ""
echo "Training DenseNet (heavy regularization)..."
python train_improved.py \
    --model dense \
    --data-dir "$DATA_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr 5e-5 \
    --weight-decay 0.1 \
    --dropout 0.5 \
    --label-smoothing 0.15 \
    --mixup 0.3 \
    --augment \
    --patience 15 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/dense.log"

# ============================================
# ImageResNet - Heavy regularization
# ============================================
echo ""
echo "Training ImageResNet (heavy regularization)..."
python train_improved.py \
    --model image_resnet \
    --data-dir "$DATA_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr 5e-5 \
    --weight-decay 0.1 \
    --dropout 0.5 \
    --label-smoothing 0.15 \
    --mixup 0.3 \
    --augment \
    --patience 15 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/image_resnet.log"

echo ""
echo "=============================================="
echo "Improved training complete!"
echo "Results in: $OUTPUT_DIR"
echo "=============================================="
echo ""
echo "Check confusion_matrix.png in each model's folder to see"
echo "which threshold classes are being confused."
