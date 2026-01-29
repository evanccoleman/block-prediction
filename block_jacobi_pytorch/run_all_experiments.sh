#!/bin/bash
# run_all_experiments.sh - Run a comprehensive set of experiments
#
# Usage: ./scripts/run_all_experiments.sh
#
# This script runs:
# 1. Basic training for all models
# 2. Model comparison (classification + regression)
# 3. Scalability study for the best model
# 4. Hyperparameter sweep for fine-tuning

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_ROOT="./experiments/full_study_$TIMESTAMP"

mkdir -p "$EXPERIMENT_ROOT"

echo "=============================================="
echo "Full Experimental Study"
echo "=============================================="
echo "Timestamp: $TIMESTAMP"
echo "Output: $EXPERIMENT_ROOT"
echo ""

# Log everything
exec > >(tee -a "$EXPERIMENT_ROOT/experiment.log") 2>&1

# 1. Quick sanity check with small dataset
echo ""
echo "========== Phase 1: Sanity Check =========="
echo ""
for model in diagonal_cnn dense image_resnet; do
    echo "Quick test: $model"
    python train.py \
        --model $model \
        --task classification \
        --synthetic \
        --num-samples 200 \
        --epochs 5 \
        --batch-size 16 \
        --output-dir "$EXPERIMENT_ROOT/sanity_check/$model"
done
echo "Sanity check passed!"

# 2. Model comparison
echo ""
echo "========== Phase 2: Model Comparison =========="
echo ""
./scripts/run_model_comparison.sh 2000 40 2>&1
mv ./experiments/model_comparison_* "$EXPERIMENT_ROOT/"

# 3. Scalability study (for the paper's original model)
echo ""
echo "========== Phase 3: Scalability Study =========="
echo ""
./scripts/run_scalability_study.sh diagonal_cnn 2>&1
mv ./experiments/scalability "$EXPERIMENT_ROOT/"

# 4. Limited hyperparameter exploration
echo ""
echo "========== Phase 4: Hyperparameter Exploration =========="
echo ""

# Just test a few key combinations
for model in diagonal_cnn dense; do
    for lr in 1e-4 5e-5; do
        for wd in 0.01 0.02; do
            RUN_DIR="$EXPERIMENT_ROOT/hyperparam/${model}_lr${lr}_wd${wd}"
            echo "Testing: $model, lr=$lr, wd=$wd"
            python train.py \
                --model $model \
                --task classification \
                --synthetic \
                --num-samples 1500 \
                --epochs 30 \
                --batch-size 16 \
                --lr $lr \
                --weight-decay $wd \
                --output-dir "$RUN_DIR"
        done
    done
done

# 5. Generate final report
echo ""
echo "========== Phase 5: Final Report =========="
echo ""

python << EOF
import json
import os
from pathlib import Path
from datetime import datetime

experiment_root = Path("$EXPERIMENT_ROOT")

report = {
    "timestamp": "$TIMESTAMP",
    "experiments": {}
}

# Collect all results
for exp_dir in experiment_root.glob("**/"):
    config_path = exp_dir / "config.json"
    history_path = exp_dir / "history.json"
    
    if not config_path.exists():
        continue
    
    with open(config_path) as f:
        config = json.load(f)
    
    result = {
        "model": config.get("model"),
        "task": config.get("task"),
        "num_samples": config.get("num_samples"),
        "epochs": config.get("epochs"),
        "lr": config.get("lr"),
        "weight_decay": config.get("weight_decay"),
    }
    
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        result["best_val_loss"] = min(history["val_loss"])
        result["final_val_loss"] = history["val_loss"][-1]
        if history.get("val_acc"):
            result["best_val_acc"] = max(history["val_acc"])
    
    rel_path = str(exp_dir.relative_to(experiment_root))
    report["experiments"][rel_path] = result

# Find best models
best_by_model = {}
for path, result in report["experiments"].items():
    model = result.get("model")
    if model and "best_val_loss" in result:
        if model not in best_by_model or result["best_val_loss"] < best_by_model[model]["loss"]:
            best_by_model[model] = {"loss": result["best_val_loss"], "path": path}

report["best_models"] = best_by_model

# Save report
with open(experiment_root / "final_report.json", "w") as f:
    json.dump(report, f, indent=2)

# Print summary
print("\n" + "="*60)
print("EXPERIMENT SUMMARY")
print("="*60)
print(f"\nTotal experiments run: {len(report['experiments'])}")
print("\nBest model for each architecture:")
print("-"*60)
for model, info in sorted(best_by_model.items()):
    print(f"  {model}: loss={info['loss']:.4f} ({info['path']})")

print(f"\nFull report saved to: {experiment_root / 'final_report.json'}")
EOF

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "Results in: $EXPERIMENT_ROOT"
echo "=============================================="
