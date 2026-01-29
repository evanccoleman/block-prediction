#!/bin/bash
# run_classification_vs_regression.sh - Compare classification and regression approaches
#
# Usage: ./scripts/run_classification_vs_regression.sh /path/to/data/directory
#
# This script trains each model architecture on both tasks:
# - Classification: Predict optimal threshold class (8 classes)
# - Regression: Predict optimal threshold fraction (continuous in [0.05, 0.40])

set -e

DATA_DIR=${1:?"Usage: $0 /path/to/data/directory"}
OUTPUT_DIR="./experiments/task_comparison"
EPOCHS=100
BATCH_SIZE=16
LR=1e-4

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Classification vs Regression Comparison"
echo "=============================================="
echo "Data directory: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Models to test
MODELS=("diagonal_cnn" "dense" "image_resnet")

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "##############################################"
    echo "# Model: $MODEL"
    echo "##############################################"
    
    # Classification
    echo ""
    echo "--- Task: Classification ---"
    python train.py \
        --model $MODEL \
        --task classification \
        --data-dir "$DATA_DIR" \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --output-dir "$OUTPUT_DIR/${MODEL}_classification" \
        2>&1 | tee "$OUTPUT_DIR/${MODEL}_classification.log"
    
    # Regression
    echo ""
    echo "--- Task: Regression ---"
    python train.py \
        --model $MODEL \
        --task regression \
        --data-dir "$DATA_DIR" \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --output-dir "$OUTPUT_DIR/${MODEL}_regression" \
        2>&1 | tee "$OUTPUT_DIR/${MODEL}_regression.log"
done

echo ""
echo "=============================================="
echo "Generating comparison summary..."
echo "=============================================="

# Generate summary
python << 'EOF'
import json
from pathlib import Path

output_dir = Path("./experiments/task_comparison")
results = []

for run_dir in sorted(output_dir.glob("*_*/")):
    if not run_dir.is_dir():
        continue
        
    # Find the actual run subdirectory
    subdirs = list(run_dir.glob("*/config.json"))
    if not subdirs:
        continue
    
    config_path = subdirs[0]
    history_path = config_path.parent / "history.json"
    
    with open(config_path) as f:
        config = json.load(f)
    
    result = {
        "model": config.get("model"),
        "task": config.get("task"),
    }
    
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        result["best_val_loss"] = min(history["val_loss"])
        result["final_val_loss"] = history["val_loss"][-1]
        
        if "val_acc" in history and history["val_acc"]:
            result["best_val_acc"] = max(history["val_acc"])
        if "val_mae" in history and history["val_mae"]:
            result["best_val_mae"] = min(history["val_mae"])
    
    results.append(result)

# Print summary
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

print("\n--- Classification Results ---")
print(f"{'Model':<15} {'Best Val Loss':<15} {'Best Val Acc':<15}")
print("-"*45)
for r in results:
    if r.get("task") == "classification":
        acc = f"{r.get('best_val_acc', 'N/A'):.4f}" if isinstance(r.get('best_val_acc'), float) else "N/A"
        print(f"{r['model']:<15} {r.get('best_val_loss', 'N/A'):<15.4f} {acc:<15}")

print("\n--- Regression Results ---")
print(f"{'Model':<15} {'Best Val Loss':<15} {'Best Val MAE':<15}")
print("-"*45)
for r in results:
    if r.get("task") == "regression":
        mae = f"{r.get('best_val_mae', 'N/A'):.4f}" if isinstance(r.get('best_val_mae'), float) else "N/A"
        print(f"{r['model']:<15} {r.get('best_val_loss', 'N/A'):<15.4f} {mae:<15}")

print("\n" + "="*80)
print("Note: MAE is in original scale [0.05, 0.40]")
print("      A MAE of 0.05 means the model is off by one class on average")
print("="*80)

# Save results
with open(output_dir / "comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)
    
print(f"\nResults saved to: {output_dir / 'comparison_results.json'}")
EOF

echo ""
echo "=============================================="
echo "Task comparison complete!"
echo "=============================================="
