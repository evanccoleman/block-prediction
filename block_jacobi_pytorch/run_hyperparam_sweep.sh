#!/bin/bash
# run_hyperparam_sweep.sh - Hyperparameter sweep for model tuning
#
# Usage: ./scripts/run_hyperparam_sweep.sh [model_name]
# Example: ./scripts/run_hyperparam_sweep.sh diagonal_cnn

set -e

MODEL=${1:-diagonal_cnn}
OUTPUT_DIR="./experiments/hyperparam_sweep/$MODEL"
EPOCHS=30
NUM_SAMPLES=2000

mkdir -p "$OUTPUT_DIR"

echo "Running hyperparameter sweep for: $MODEL"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Learning rates to try
LEARNING_RATES=(1e-3 5e-4 1e-4 5e-5 1e-5)

# Batch sizes to try
BATCH_SIZES=(8 16 32)

# Weight decay values
WEIGHT_DECAYS=(0.0 0.01 0.02 0.05)

# Run experiments
for lr in "${LEARNING_RATES[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        for wd in "${WEIGHT_DECAYS[@]}"; do
            
            RUN_NAME="lr${lr}_bs${bs}_wd${wd}"
            RUN_DIR="$OUTPUT_DIR/$RUN_NAME"
            
            # Skip if already completed
            if [ -f "$RUN_DIR/best_model.pt" ]; then
                echo "Skipping $RUN_NAME (already complete)"
                continue
            fi
            
            echo "----------------------------------------"
            echo "Running: lr=$lr, batch_size=$bs, weight_decay=$wd"
            echo "----------------------------------------"
            
            python train.py \
                --model $MODEL \
                --task classification \
                --synthetic \
                --num-samples $NUM_SAMPLES \
                --epochs $EPOCHS \
                --batch-size $bs \
                --lr $lr \
                --weight-decay $wd \
                --output-dir "$RUN_DIR" \
                2>&1 | tee "$RUN_DIR.log"
            
            echo ""
        done
    done
done

echo "=============================================="
echo "Hyperparameter sweep complete!"
echo "=============================================="

# Summarize results
echo ""
echo "Summary of best validation losses:"
echo "-----------------------------------"
for dir in "$OUTPUT_DIR"/*/; do
    if [ -f "$dir/config.json" ]; then
        name=$(basename "$dir")
        # Extract best val loss from the last line of training
        if [ -f "${dir%/}.log" ]; then
            best_loss=$(grep -oP "Val Loss: \K[0-9.]+" "${dir%/}.log" | sort -n | head -1)
            echo "$name: $best_loss"
        fi
    fi
done
