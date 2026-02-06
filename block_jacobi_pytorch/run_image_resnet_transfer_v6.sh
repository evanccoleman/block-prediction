#!/bin/bash
# run_image_resnet_transfer_v6.sh - Transfer evaluation with ALL CORRECT APIs
#
# Model API (ImageResNet):
#   n_size, in_channels=1, num_classes=8, task, dropout=0.5
#
# Dataset API (BlockJacobiDataset):
#   data_dir, input_type='diagonal'|'matrix'|'image', task, ...
#
# For ImageResNet: input_type='image' (uses PNG files)
#
# Usage: ./run_image_resnet_transfer_v6.sh
# Run from block_jacobi_pytorch directory

set -e

OUTPUT_DIR="./experiments/transfer_study"
RESULTS_CSV="$OUTPUT_DIR/image_resnet_transfer_results.csv"

# Find the trained image_resnet model
MODEL_PATH=$(find "$OUTPUT_DIR/trained_image_resnet" -name "best_model.pt" -type f 2>/dev/null | head -1)

if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: No trained image_resnet model found"
    exit 1
fi

echo "=============================================="
echo "ImageResNet Transfer Learning Study (v6)"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo ""

# Initialize results
echo "test_size,accuracy,loss" > "$RESULTS_CSV"

for TEST_DIR in ../../dataset_128 ../../dataset_500 ../../dataset_1000 ../../dataset_2500 ../../dataset_5000 ../../dataset_10000; do
    if [ ! -d "$TEST_DIR" ]; then
        echo "Skipping $TEST_DIR (not found)"
        continue
    fi
    
    # Get matrix size
    TEST_JSON=$(find "$TEST_DIR" -name "*.json" -type f 2>/dev/null | head -1)
    if [ -n "$TEST_JSON" ]; then
        TEST_SIZE=$(python3 -c "import json; d=json.load(open('$TEST_JSON')); print(d.get('matrix_properties',{}).get('size', 'unknown'))" 2>/dev/null || echo "unknown")
    else
        TEST_SIZE="unknown"
    fi
    
    echo ""
    echo "Testing on n=$TEST_SIZE..."
    
    python3 << PYEOF
import torch
import json
import sys
from pathlib import Path
sys.path.insert(0, '.')

MODEL_PATH = '$MODEL_PATH'
TEST_DIR = '$TEST_DIR'
TEST_SIZE = '$TEST_SIZE'

# Load config to get training parameters
config_path = Path(MODEL_PATH).parent / 'config.json'
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    print(f"Config: n_size={config.get('n_size')}, task={config.get('task')}")
else:
    config = {}
    print("No config found, using defaults")

# Get training parameters from config
train_n_size = config.get('n_size', config.get('matrix_size', 128))
num_classes = config.get('num_classes', 8)
task = config.get('task', 'classification')

print(f"Creating ImageResNet with n_size={train_n_size}, num_classes={num_classes}, task={task}")

# Create model with CORRECT API
from models import ImageResNet
model = ImageResNet(
    n_size=train_n_size,
    in_channels=1,
    num_classes=num_classes,
    task=task
)

# Load weights (handle both dict and raw state_dict formats)
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
print("Weights loaded successfully")

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

# Load test data with CORRECT API
# BlockJacobiDataset signature:
#   data_dir, input_type='diagonal'|'matrix'|'image', task, ...
# For ImageResNet, use input_type='image'
from data import BlockJacobiDataset
from torch.utils.data import DataLoader

try:
    test_dataset = BlockJacobiDataset(
        data_dir=TEST_DIR,
        input_type='image',  # CORRECT: use 'image' for ImageResNet
        task=task
    )
    print(f"Loaded {len(test_dataset)} test samples")
except Exception as e:
    print(f"Error loading data: {e}")
    import traceback
    traceback.print_exc()
    print(f"RESULT,{TEST_SIZE},N/A,N/A")
    sys.exit(0)

if len(test_dataset) == 0:
    print("No samples loaded!")
    print(f"RESULT,{TEST_SIZE},N/A,N/A")
    sys.exit(0)

# Verify sample shape
sample_input, sample_label = test_dataset[0]
print(f"Sample input shape: {sample_input.shape}, label: {sample_label}")

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Evaluate
correct = 0
total = 0
total_loss = 0.0
criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

if total > 0:
    accuracy = correct / total
    avg_loss = total_loss / total
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"RESULT,{TEST_SIZE},{accuracy:.4f},{avg_loss:.4f}")
else:
    print(f"RESULT,{TEST_SIZE},N/A,N/A")
PYEOF
    
done | tee "$OUTPUT_DIR/image_resnet_transfer_v6.log"

# Extract results
grep "^RESULT," "$OUTPUT_DIR/image_resnet_transfer_v6.log" | sed 's/RESULT,//' >> "$RESULTS_CSV"

echo ""
echo "=============================================="
echo "FINAL RESULTS"
echo "=============================================="
cat "$RESULTS_CSV" | column -t -s','
echo ""
echo "Saved to: $RESULTS_CSV"
