#!/bin/bash
# run_model_comparison.sh - Compare all models under identical conditions
#
# Usage: ./scripts/run_model_comparison.sh [num_samples] [epochs]
# Example: ./scripts/run_model_comparison.sh 3000 50

set -e

NUM_SAMPLES=${1:-3000}
EPOCHS=${2:-50}
BATCH_SIZE=16
LR=1e-4
SEED=42
OUTPUT_DIR="./experiments/model_comparison_n${NUM_SAMPLES}_e${EPOCHS}"

mkdir -p "$OUTPUT_DIR"

MODELS=("diagonal_cnn" "dense" "image_cnn" "image_resnet")
TASKS=("classification" "regression")

echo "=============================================="
echo "Model Comparison Experiment"
echo "=============================================="
echo "Samples: $NUM_SAMPLES"
echo "Epochs: $EPOCHS"
echo "Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo ""

for task in "${TASKS[@]}"; do
    echo ""
    echo "########## Task: $task ##########"
    echo ""
    
    for model in "${MODELS[@]}"; do
        RUN_NAME="${model}_${task}"
        RUN_DIR="$OUTPUT_DIR/$RUN_NAME"
        
        if [ -f "$RUN_DIR/best_model.pt" ]; then
            echo "Skipping $RUN_NAME (already complete)"
            continue
        fi
        
        echo "Training: $model ($task)"
        echo "-----------------------------------"
        
        python train.py \
            --model $model \
            --task $task \
            --synthetic \
            --num-samples $NUM_SAMPLES \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --lr $LR \
            --seed $SEED \
            --output-dir "$RUN_DIR" \
            2>&1 | tee "$RUN_DIR.log"
        
        echo ""
    done
done

echo "=============================================="
echo "Generating comparison report..."
echo "=============================================="

# Generate summary report
python << EOF
import json
import os
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")
results = []

for run_dir in sorted(output_dir.glob("*/")):
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.json"
    
    if not config_path.exists():
        continue
    
    with open(config_path) as f:
        config = json.load(f)
    
    result = {
        "model": config["model"],
        "task": config["task"],
    }
    
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        result["final_val_loss"] = history["val_loss"][-1]
        result["best_val_loss"] = min(history["val_loss"])
        if "val_acc" in history and history["val_acc"]:
            result["final_val_acc"] = history["val_acc"][-1]
            result["best_val_acc"] = max(history["val_acc"])
    
    results.append(result)

# Print table
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"{'Model':<15} {'Task':<15} {'Best Loss':<12} {'Final Loss':<12} {'Best Acc':<10}")
print("-"*70)

for r in results:
    acc = f"{r.get('best_val_acc', 'N/A'):.4f}" if isinstance(r.get('best_val_acc'), float) else "N/A"
    print(f"{r['model']:<15} {r['task']:<15} {r.get('best_val_loss', 'N/A'):<12.4f} "
          f"{r.get('final_val_loss', 'N/A'):<12.4f} {acc:<10}")

# Save to JSON
with open(output_dir / "comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to:", output_dir / "comparison_results.json")
EOF

echo ""
echo "Model comparison complete!"
