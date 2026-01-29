#!/bin/bash
# run_real_data.sh - Train models on your real matrix data
#
# Usage: ./scripts/run_real_data.sh /path/to/data/directory
#
# Supports two data formats:
#
# 1. Flat structure:
#    data_dir/matrix_0.npz, matrix_0.json, matrix_0.png
#
# 2. Subfolder structure (png_builder2.py format):
#    data_dir/
#      images/matrix_0.png, matrix_1.png, ...
#      matrices/matrix_0.npz, matrix_1.npz, ... (optional, needs --save_raw)
#      metadata/matrix_0.json, matrix_1.json, ...

set -e

DATA_DIR=${1:?"Usage: $0 /path/to/data/directory"}
OUTPUT_DIR="./experiments/real_data"
EPOCHS=100
BATCH_SIZE=8
LR=1e-4

# Verify data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

# Detect folder structure
if [ -d "$DATA_DIR/metadata" ]; then
    echo "Detected subfolder structure (png_builder2.py format)"
    STRUCTURE="subfolder"
    NUM_SAMPLES=$(ls -1 "$DATA_DIR/metadata"/matrix_*.json 2>/dev/null | wc -l)
    NUM_MATRICES=$(ls -1 "$DATA_DIR/matrices"/matrix_*.npz 2>/dev/null | wc -l)
    NUM_PNGS=$(ls -1 "$DATA_DIR/images"/matrix_*.png 2>/dev/null | wc -l)
else
    echo "Detected flat structure"
    STRUCTURE="flat"
    NUM_SAMPLES=$(ls -1 "$DATA_DIR"/matrix_*.json 2>/dev/null | wc -l)
    NUM_MATRICES=$(ls -1 "$DATA_DIR"/matrix_*.npz 2>/dev/null | wc -l)
    NUM_PNGS=$(ls -1 "$DATA_DIR"/matrix_*.png 2>/dev/null | wc -l)
fi

if [ "$NUM_SAMPLES" -eq 0 ]; then
    echo "Error: No metadata files found in $DATA_DIR"
    exit 1
fi

HAS_MATRICES=$( [ "$NUM_MATRICES" -gt 0 ] && echo "yes" || echo "no" )
HAS_IMAGES=$( [ "$NUM_PNGS" -gt 0 ] && echo "yes" || echo "no" )

echo "=============================================="
echo "Training on Real Data"
echo "=============================================="
echo "Data directory: $DATA_DIR"
echo "Structure: $STRUCTURE"
echo "Samples (metadata): $NUM_SAMPLES"
echo "Raw matrices (.npz): $NUM_MATRICES ($HAS_MATRICES)"
echo "PNG images: $NUM_PNGS ($HAS_IMAGES)"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Train models based on available data
if [ "$HAS_MATRICES" = "yes" ]; then
    # Train DiagonalCNN (requires raw matrices)
    echo "Training DiagonalCNN..."
    python train.py \
        --model diagonal_cnn \
        --task classification \
        --data-dir "$DATA_DIR" \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --output-dir "$OUTPUT_DIR/diagonal_cnn" \
        2>&1 | tee "$OUTPUT_DIR/diagonal_cnn.log"

    # Train DenseNet (requires raw matrices)
    echo ""
    echo "Training DenseNet..."
    python train.py \
        --model dense \
        --task classification \
        --data-dir "$DATA_DIR" \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --output-dir "$OUTPUT_DIR/dense" \
        2>&1 | tee "$OUTPUT_DIR/dense.log"
else
    echo "Skipping DiagonalCNN and DenseNet (no .npz files)"
    echo "  Hint: Use --save_raw flag when running png_builder2.py"
fi

# Train ImageResNet if images are available
if [ "$HAS_IMAGES" = "yes" ]; then
    echo ""
    echo "Training ImageResNet..."
    python train.py \
        --model image_resnet \
        --task classification \
        --data-dir "$DATA_DIR" \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --output-dir "$OUTPUT_DIR/image_resnet" \
        2>&1 | tee "$OUTPUT_DIR/image_resnet.log"
else
    echo ""
    echo "Skipping ImageResNet (no PNG files available)"
fi

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
