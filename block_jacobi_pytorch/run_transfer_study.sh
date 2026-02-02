#!/bin/bash
# run_transfer_study.sh - Test generalization across matrix sizes
#
# Usage: ./scripts/run_transfer_study.sh /path/to/train_data/ /path/to/test_data1/ /path/to/test_data2/ ...
#
# This tests the key claim: models trained on small matrices generalize to larger ones.
# 
# Example:
#   ./scripts/run_transfer_study.sh dataset_128/ dataset_500/ dataset_2000/ dataset_10000/

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 /path/to/train_data/ /path/to/test_data1/ [/path/to/test_data2/ ...]"
    echo ""
    echo "Example: $0 dataset_128/ dataset_500/ dataset_2000/ dataset_10000/"
    exit 1
fi

TRAIN_DIR="$1"
shift
TEST_DIRS=("$@")

EPOCHS=${EPOCHS:-30}
OUTPUT_DIR="./experiments/transfer_study"
BATCH_SIZE=16
LR=1e-4

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "TRANSFER LEARNING STUDY"
echo "=============================================="
echo "Training data: $TRAIN_DIR"
echo "Test datasets: ${TEST_DIRS[*]}"
echo "Epochs: $EPOCHS"
echo ""

# Detect training matrix size
TRAIN_JSON=$(find "$TRAIN_DIR" -name "*.json" -type f 2>/dev/null | head -1)
if [ -n "$TRAIN_JSON" ]; then
    TRAIN_SIZE=$(python3 -c "import json; d=json.load(open('$TRAIN_JSON')); print(d.get('matrix_properties',{}).get('size', 'unknown'))" 2>/dev/null || echo "unknown")
    echo "Training matrix size: ${TRAIN_SIZE}x${TRAIN_SIZE}"
else
    TRAIN_SIZE="unknown"
fi
echo ""

# Models to test (only scalable ones for transfer)
MODELS=("diagonal_cnn" "scalable_diagonal" "conv_dense" "image_resnet")

# Results CSV
RESULTS_CSV="$OUTPUT_DIR/transfer_results.csv"
echo "model,train_size,test_size,test_accuracy,test_loss" > "$RESULTS_CSV"

# ============================================
# PHASE 1: Train on source dataset
# ============================================
echo "=============================================="
echo "PHASE 1: Training models on $TRAIN_DIR"
echo "=============================================="

for MODEL in "${MODELS[@]}"; do
    TRAIN_OUTPUT="$OUTPUT_DIR/trained_${MODEL}"
    
    if [ -f "$TRAIN_OUTPUT/best_model.pt" ]; then
        echo "  $MODEL: already trained, skipping"
        continue
    fi
    
    echo "  Training $MODEL..."
    python3 train.py \
        --model "$MODEL" \
        --task classification \
        --data-dir "$TRAIN_DIR" \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --output-dir "$TRAIN_OUTPUT" \
        2>&1 | tee "$TRAIN_OUTPUT.log"
    
    # Record training accuracy
    TRAIN_ACC=$(grep -oP "Val Acc: \K[\d.]+" "$TRAIN_OUTPUT.log" 2>/dev/null | sort -rn | head -1 || echo "N/A")
    TRAIN_LOSS=$(grep -oP "Best validation loss: \K[\d.]+" "$TRAIN_OUTPUT.log" 2>/dev/null | tail -1 || echo "N/A")
    echo "$MODEL,$TRAIN_SIZE,$TRAIN_SIZE,$TRAIN_ACC,$TRAIN_LOSS" >> "$RESULTS_CSV"
done

# ============================================
# PHASE 2: Evaluate on target datasets
# ============================================
echo ""
echo "=============================================="
echo "PHASE 2: Evaluating on target datasets"
echo "=============================================="

for TEST_DIR in "${TEST_DIRS[@]}"; do
    # Detect test matrix size
    TEST_JSON=$(find "$TEST_DIR" -name "*.json" -type f 2>/dev/null | head -1)
    if [ -n "$TEST_JSON" ]; then
        TEST_SIZE=$(python3 -c "import json; d=json.load(open('$TEST_JSON')); print(d.get('matrix_properties',{}).get('size', 'unknown'))" 2>/dev/null || echo "unknown")
    else
        TEST_SIZE="unknown"
    fi
    
    echo ""
    echo "Testing on: $TEST_DIR (size: ${TEST_SIZE}x${TEST_SIZE})"
    echo "----------------------------------------------"
    
    for MODEL in "${MODELS[@]}"; do
        TRAIN_OUTPUT="$OUTPUT_DIR/trained_${MODEL}"
        MODEL_PATH="$TRAIN_OUTPUT/best_model.pt"
        
        if [ ! -f "$MODEL_PATH" ]; then
            echo "  $MODEL: no trained model found, skipping"
            continue
        fi
        
        echo "  Evaluating $MODEL..."
        
        # Run evaluation
        EVAL_LOG="$OUTPUT_DIR/eval_${MODEL}_on_${TEST_SIZE}.log"
        
        python3 -c "
import torch
import json
from pathlib import Path
import sys
sys.path.insert(0, '.')

from models import create_model
from data import BlockJacobiDataset
from torch.utils.data import DataLoader

# Load config to get model parameters
config_path = Path('$TRAIN_OUTPUT') / 'config.json'
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
else:
    config = {}

# Create model
model = create_model(
    '$MODEL',
    task='classification',
    n_classes=8,
    matrix_size=config.get('matrix_size', 128)
)

# Load weights
model.load_state_dict(torch.load('$MODEL_PATH', map_location='cpu'))
model.eval()

# Load test data
test_dataset = BlockJacobiDataset(
    data_dir='$TEST_DIR',
    task='classification',
    model_type='$MODEL'
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate
correct = 0
total = 0
total_loss = 0.0
criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

accuracy = correct / total
avg_loss = total_loss / total

print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Loss: {avg_loss:.4f}')
" 2>&1 | tee "$EVAL_LOG"
        
        # Extract results
        TEST_ACC=$(grep -oP "Test Accuracy: \K[\d.]+" "$EVAL_LOG" 2>/dev/null || echo "N/A")
        TEST_LOSS=$(grep -oP "Test Loss: \K[\d.]+" "$EVAL_LOG" 2>/dev/null || echo "N/A")
        
        echo "$MODEL,$TRAIN_SIZE,$TEST_SIZE,$TEST_ACC,$TEST_LOSS" >> "$RESULTS_CSV"
        echo "    → Accuracy: $TEST_ACC, Loss: $TEST_LOSS"
    done
done

# ============================================
# Generate Summary Report
# ============================================
echo ""
echo "=============================================="
echo "TRANSFER LEARNING SUMMARY"
echo "=============================================="

python3 << 'PYTHON_SCRIPT'
import pandas as pd
from pathlib import Path

results_file = Path("./experiments/transfer_study/transfer_results.csv")
df = pd.read_csv(results_file)

print("\nTransfer Results (trained on smallest size):")
print("=" * 70)

# Pivot table: models as rows, test sizes as columns
pivot = df.pivot_table(
    index='model', 
    columns='test_size', 
    values='test_accuracy',
    aggfunc='first'
)

# Sort columns numerically if possible
try:
    pivot = pivot.reindex(sorted(pivot.columns, key=lambda x: int(x) if str(x).isdigit() else 0), axis=1)
except:
    pass

print(pivot.to_string())

# Calculate degradation
print("\n\nAccuracy Degradation from Training Size:")
print("-" * 70)

train_size = df['train_size'].iloc[0]
for model in df['model'].unique():
    model_df = df[df['model'] == model]
    train_acc = model_df[model_df['test_size'] == train_size]['test_accuracy'].values
    if len(train_acc) > 0:
        train_acc = float(train_acc[0]) if train_acc[0] != 'N/A' else None
        if train_acc:
            print(f"\n{model}:")
            for _, row in model_df.iterrows():
                if row['test_accuracy'] != 'N/A' and row['test_size'] != train_size:
                    test_acc = float(row['test_accuracy'])
                    degradation = train_acc - test_acc
                    print(f"  {train_size} → {row['test_size']}: {test_acc:.4f} (Δ = {degradation:+.4f})")

print("\n" + "=" * 70)
print("Results saved to: ./experiments/transfer_study/transfer_results.csv")
PYTHON_SCRIPT

echo ""
echo "Transfer study complete!"
