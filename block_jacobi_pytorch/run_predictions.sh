#!/bin/bash
# run_predictions.sh - Run predictions on a directory of matrices
#
# Usage: ./scripts/run_predictions.sh /path/to/model/best_model.pt /path/to/matrices/

set -e

MODEL_PATH=${1:?"Usage: $0 /path/to/model/best_model.pt /path/to/matrices/"}
INPUT_DIR=${2:?"Usage: $0 /path/to/model/best_model.pt /path/to/matrices/"}

# Verify paths
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Derive output path
MODEL_DIR=$(dirname "$MODEL_PATH")
OUTPUT_FILE="$MODEL_DIR/predictions_$(basename "$INPUT_DIR").json"

echo "=============================================="
echo "Running Predictions"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_FILE"
echo ""

python predict.py \
    --model-path "$MODEL_PATH" \
    --input-dir "$INPUT_DIR" \
    --output "$OUTPUT_FILE"

echo ""
echo "Predictions saved to: $OUTPUT_FILE"
