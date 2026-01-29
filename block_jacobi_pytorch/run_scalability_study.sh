#!/bin/bash
# run_scalability_study.sh - Study how models scale with dataset size
#
# Usage: ./scripts/run_scalability_study.sh [model]
# Example: ./scripts/run_scalability_study.sh diagonal_cnn

set -e

MODEL=${1:-diagonal_cnn}
OUTPUT_DIR="./experiments/scalability/$MODEL"
EPOCHS=30
BATCH_SIZE=16
SEED=42

mkdir -p "$OUTPUT_DIR"

# Dataset sizes to test
SAMPLE_SIZES=(100 250 500 1000 2000 3000 5000)

echo "=============================================="
echo "Scalability Study: $MODEL"
echo "=============================================="
echo "Sample sizes: ${SAMPLE_SIZES[*]}"
echo ""

for n in "${SAMPLE_SIZES[@]}"; do
    RUN_NAME="samples_$n"
    RUN_DIR="$OUTPUT_DIR/$RUN_NAME"
    
    if [ -f "$RUN_DIR/best_model.pt" ]; then
        echo "Skipping n=$n (already complete)"
        continue
    fi
    
    echo "Training with $n samples..."
    
    START_TIME=$(date +%s)
    
    python train.py \
        --model $MODEL \
        --task classification \
        --synthetic \
        --num-samples $n \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --seed $SEED \
        --output-dir "$RUN_DIR" \
        2>&1 | tee "$RUN_DIR.log"
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo "  Completed in ${ELAPSED}s"
    echo ""
done

# Generate scalability report
echo "=============================================="
echo "Generating scalability report..."
echo "=============================================="

python << EOF
import json
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")
sample_sizes = []
best_losses = []
best_accs = []

for run_dir in sorted(output_dir.glob("samples_*/")):
    n = int(run_dir.name.split("_")[1])
    history_path = run_dir / "history.json"
    
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        
        sample_sizes.append(n)
        best_losses.append(min(history["val_loss"]))
        if history.get("val_acc"):
            best_accs.append(max(history["val_acc"]))

# Print table
print("\nScalability Results for $MODEL")
print("="*50)
print(f"{'Samples':<10} {'Best Val Loss':<15} {'Best Val Acc':<15}")
print("-"*50)
for i, n in enumerate(sample_sizes):
    acc = f"{best_accs[i]:.4f}" if i < len(best_accs) else "N/A"
    print(f"{n:<10} {best_losses[i]:<15.4f} {acc:<15}")

# Create plot
if len(sample_sizes) > 1:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(sample_sizes, best_losses, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Training Samples')
    ax1.set_ylabel('Best Validation Loss')
    ax1.set_title('$MODEL: Loss vs Dataset Size')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    if best_accs:
        ax2.plot(sample_sizes[:len(best_accs)], best_accs, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Training Samples')
        ax2.set_ylabel('Best Validation Accuracy')
        ax2.set_title('$MODEL: Accuracy vs Dataset Size')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "scalability_plot.png", dpi=150)
    print(f"\nPlot saved to: {output_dir / 'scalability_plot.png'}")

# Save data
results = {
    "model": "$MODEL",
    "sample_sizes": sample_sizes,
    "best_val_losses": best_losses,
    "best_val_accs": best_accs,
}
with open(output_dir / "scalability_results.json", "w") as f:
    json.dump(results, f, indent=2)
EOF

echo ""
echo "Scalability study complete!"
