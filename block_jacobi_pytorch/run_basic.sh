#!/bin/bash
# run_basic.sh - Basic training examples for each model architecture
#
# Usage: ./scripts/run_basic.sh

set -e  # Exit on error

# Configuration
EPOCHS=50
BATCH_SIZE=16
OUTPUT_DIR="./experiments/basic"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Training DiagonalCNN (Götz & Anzt approach)"
echo "=============================================="
python train.py \
    --model diagonal_cnn \
    --task classification \
    --synthetic \
    --num-samples 3000 \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr 1e-4 \
    --output-dir "$OUTPUT_DIR/diagonal_cnn"

echo ""
echo "=============================================="
echo "Training DenseNet (full matrix approach)"
echo "=============================================="
python train.py \
    --model dense \
    --task classification \
    --synthetic \
    --num-samples 3000 \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr 1e-4 \
    --output-dir "$OUTPUT_DIR/dense"

echo ""
echo "=============================================="
echo "Training ImageResNet (image-based approach)"
echo "=============================================="
python train.py \
    --model image_resnet \
    --task classification \
    --synthetic \
    --num-samples 3000 \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr 1e-4 \
    --output-dir "$OUTPUT_DIR/image_resnet"

echo ""
echo "=============================================="
echo "All basic experiments complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
