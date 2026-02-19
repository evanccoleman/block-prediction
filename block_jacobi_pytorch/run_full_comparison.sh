#!/bin/bash
# run_full_comparison.sh - Compare ALL model architectures
#
# Usage: ./scripts/run_full_comparison.sh /path/to/data/directory [epochs] [output_name]
#
# Examples:
#   ./scripts/run_full_comparison.sh dataset_128/ 50
#   # → saves to experiments/full_comparison_n128/
#
#   ./scripts/run_full_comparison.sh dataset_500/ 50 my_experiment
#   # → saves to experiments/my_experiment/
#
# This script compares:
# 1. DiagonalCNN (original paper architecture, global prediction)
# 2. ScalableDiagonalCNN (O(1) parameters version)
# 3. ConvDenseNet (memory-efficient dense alternative)
# 4. ImageResNet (image-based approach)
# 5. BlockStartCNN (per-entry prediction, closest to original paper)
#
# Each model is tested on both classification and regression tasks
# (except BlockStartCNN which is binary classification only)

set -e

DATA_DIR=${1:?"Usage: $0 /path/to/data/directory [epochs] [output_name]"}
EPOCHS=${2:-30}
OUTPUT_NAME=${3:-""}  # Optional: custom output name
BATCH_SIZE=16
LR=1e-4

# Detect matrix size first (needed for auto-naming)
FIRST_JSON=$(find "$DATA_DIR" -name "*.json" -type f 2>/dev/null | head -1)
if [ -n "$FIRST_JSON" ]; then
    N_SIZE=$(python3 -c "import json; d=json.load(open('$FIRST_JSON')); print(d.get('matrix_properties',{}).get('size', 128))" 2>/dev/null || echo "128")
else
    N_SIZE=128
fi

# Set output directory: use custom name, or auto-generate from matrix size
if [ -n "$OUTPUT_NAME" ]; then
    OUTPUT_DIR="./experiments/$OUTPUT_NAME"
else
    OUTPUT_DIR="./experiments/full_comparison_n${N_SIZE}"
fi

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "COMPREHENSIVE MODEL COMPARISON"
echo "=============================================="
echo "Data directory: $DATA_DIR"
echo "Matrix size: ${N_SIZE}x${N_SIZE}"
echo "Epochs: $EPOCHS"
echo "Output: $OUTPUT_DIR"
echo ""

# Results file
RESULTS_FILE="$OUTPUT_DIR/comparison_results.txt"
echo "MODEL COMPARISON RESULTS" > "$RESULTS_FILE"
echo "========================" >> "$RESULTS_FILE"
echo "Data: $DATA_DIR" >> "$RESULTS_FILE"
echo "Matrix size: ${N_SIZE}x${N_SIZE}" >> "$RESULTS_FILE"
echo "Epochs: $EPOCHS" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# CSV for easy parsing
CSV_FILE="$OUTPUT_DIR/comparison_results.csv"
echo "model,task,val_loss,val_metric,metric_name,params" > "$CSV_FILE"

# ============================================
# GLOBAL PREDICTION MODELS
# ============================================
#GLOBAL_MODELS=("diagonal_cnn" "scalable_diagonal" "conv_dense" "image_resnet")
GLOBAL_MODELS=("image_resnet")

for MODEL in "${GLOBAL_MODELS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Training: $MODEL"
    echo "=============================================="
    
    # Classification
    echo "  → Classification task..."
    LOG_FILE="$OUTPUT_DIR/${MODEL}_classification.log"
    
    python3 train.py \
        --model "$MODEL" \
        --task classification \
        --data-dir "$DATA_DIR" \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --output-dir "$OUTPUT_DIR/${MODEL}_classification" \
        2>&1 | tee "$LOG_FILE"
    
    # Extract metrics
    VAL_LOSS=$(grep -oP "Best validation loss: \K[\d.]+" "$LOG_FILE" 2>/dev/null | tail -1 || echo "N/A")
    VAL_ACC=$(grep -oP "Val Acc: \K[\d.]+" "$LOG_FILE" 2>/dev/null | sort -rn | head -1 || echo "N/A")
    PARAMS=$(grep -oP "Parameters: \K[\d,]+" "$LOG_FILE" 2>/dev/null | head -1 | tr -d ',' || echo "N/A")
    
    echo "$MODEL,classification,$VAL_LOSS,$VAL_ACC,accuracy,$PARAMS" >> "$CSV_FILE"
    echo "" >> "$RESULTS_FILE"
    echo "$MODEL (Classification):" >> "$RESULTS_FILE"
    echo "  Val Loss: $VAL_LOSS, Accuracy: $VAL_ACC, Params: $PARAMS" >> "$RESULTS_FILE"
    
    # Regression
    echo "  → Regression task..."
    LOG_FILE="$OUTPUT_DIR/${MODEL}_regression.log"
    
    python3 train.py \
        --model "$MODEL" \
        --task regression \
        --data-dir "$DATA_DIR" \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --output-dir "$OUTPUT_DIR/${MODEL}_regression" \
        2>&1 | tee "$LOG_FILE"
    
    # Extract metrics
    VAL_LOSS=$(grep -oP "Best validation loss: \K[\d.]+" "$LOG_FILE" 2>/dev/null | tail -1 || echo "N/A")
    VAL_MAE=$(grep -oP "Val MAE \(orig\): \K[\d.]+" "$LOG_FILE" 2>/dev/null | sort -n | head -1 || echo "N/A")
    
    echo "$MODEL,regression,$VAL_LOSS,$VAL_MAE,mae,$PARAMS" >> "$CSV_FILE"
    echo "$MODEL (Regression):" >> "$RESULTS_FILE"
    echo "  Val Loss: $VAL_LOSS, MAE: $VAL_MAE" >> "$RESULTS_FILE"
done

# ============================================
# BLOCK START CNN (Per-Entry Prediction)
# ============================================
echo ""
echo "=============================================="
echo "Training: BlockStartCNN (Per-Entry Prediction)"
echo "=============================================="
echo "  This is the original paper's formulation"

LOG_FILE="$OUTPUT_DIR/block_start_cnn.log"

python3 block_start_cnn.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR/block_start_cnn" \
    --epochs $EPOCHS \
    --batch-size 64 \
    --lr 1e-3 \
    --samples-per-matrix 200 \
    2>&1 | tee "$LOG_FILE"

# Extract metrics
VAL_LOSS=$(grep -oP "Best validation loss: \K[\d.]+" "$LOG_FILE" 2>/dev/null | tail -1 || echo "N/A")
VAL_F1=$(grep -oP "F1: \K[\d.]+" "$LOG_FILE" 2>/dev/null | sort -rn | head -1 || echo "N/A")
VAL_ACC=$(grep -oP "Acc: \K[\d.]+" "$LOG_FILE" 2>/dev/null | sort -rn | head -1 || echo "N/A")
PARAMS=$(grep -oP "Model parameters: \K[\d,]+" "$LOG_FILE" 2>/dev/null | head -1 | tr -d ',' || echo "N/A")

echo "block_start_cnn,per_entry_binary,$VAL_LOSS,$VAL_F1,f1_score,$PARAMS" >> "$CSV_FILE"
echo "" >> "$RESULTS_FILE"
echo "block_start_cnn (Per-Entry Binary Classification):" >> "$RESULTS_FILE"
echo "  Val Loss: $VAL_LOSS, F1: $VAL_F1, Accuracy: $VAL_ACC, Params: $PARAMS" >> "$RESULTS_FILE"

# ============================================
# Generate Summary
# ============================================
echo ""
echo "=============================================="
echo "SUMMARY"
echo "=============================================="

cat << 'EOF' >> "$RESULTS_FILE"

============================================
MODEL COMPARISON SUMMARY
============================================

| Model              | Task           | Params Scale | Best For                    |
|--------------------|----------------|--------------|-----------------------------| 
| diagonal_cnn       | Global (8-way) | O(n)         | Small matrices, high acc    |
| scalable_diagonal  | Global (8-way) | O(1)         | Large matrices, good acc    |
| conv_dense         | Global (8-way) | O(1)         | Full matrix context         |
| image_resnet       | Global (8-way) | O(1)         | Image-based, any size       |
| block_start_cnn    | Per-entry (2)  | O(1)         | Variable block sizes        |

SCALABILITY NOTES:
- diagonal_cnn: FC layer has 128*n parameters → fails for n > 10^4
- scalable_diagonal: Uses adaptive pooling → works for any n
- conv_dense: Uses conv compression → works for any n  
- image_resnet: Uses adaptive pooling → works for any n (image must fit in memory)
- block_start_cnn: Fixed window size → O(1) params, O(n) inference

RECOMMENDATION:
- For uniform block structure: scalable_diagonal or image_resnet
- For variable block structure: block_start_cnn (original paper approach)
- For small matrices where accuracy matters most: diagonal_cnn
EOF

echo ""
echo "Results saved to:"
echo "  - $RESULTS_FILE (human readable)"
echo "  - $CSV_FILE (machine readable)"
echo ""

# Print summary table
echo "=============================================="
echo "CLASSIFICATION RESULTS"
echo "=============================================="
printf "%-20s %-12s %-12s %-15s\n" "Model" "Val Loss" "Accuracy" "Parameters"
printf "%-20s %-12s %-12s %-15s\n" "-----" "--------" "--------" "----------"
grep "classification" "$CSV_FILE" | while IFS=',' read -r model task loss metric name params; do
    printf "%-20s %-12s %-12s %-15s\n" "$model" "$loss" "$metric" "$params"
done

echo ""
echo "=============================================="
echo "REGRESSION RESULTS"
echo "=============================================="
printf "%-20s %-12s %-12s %-15s\n" "Model" "Val Loss" "MAE" "Parameters"
printf "%-20s %-12s %-12s %-15s\n" "-----" "--------" "---" "----------"
grep "regression" "$CSV_FILE" | while IFS=',' read -r model task loss metric name params; do
    printf "%-20s %-12s %-12s %-15s\n" "$model" "$loss" "$metric" "$params"
done

echo ""
echo "=============================================="
echo "PER-ENTRY PREDICTION (BlockStartCNN)"
echo "=============================================="
printf "%-20s %-12s %-12s %-15s\n" "Model" "Val Loss" "F1 Score" "Parameters"
printf "%-20s %-12s %-12s %-15s\n" "-----" "--------" "--------" "----------"
grep "per_entry" "$CSV_FILE" | while IFS=',' read -r model task loss metric name params; do
    printf "%-20s %-12s %-12s %-15s\n" "$model" "$loss" "$metric" "$params"
done

echo ""
echo "Full comparison complete!"
