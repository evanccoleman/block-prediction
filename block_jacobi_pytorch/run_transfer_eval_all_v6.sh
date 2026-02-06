#!/bin/bash
# run_transfer_eval_all_v6.sh - Evaluate ALL trained models with CORRECT APIs
#
# Model → input_type mapping:
#   diagonal_cnn      → input_type='diagonal'  (needs .npz files)
#   scalable_diagonal → input_type='matrix'    (needs .npz files)  
#   conv_dense        → input_type='matrix'    (needs .npz files)
#   image_resnet      → input_type='image'     (uses .png files)
#
# Usage: ./run_transfer_eval_all_v6.sh
# Run from block_jacobi_pytorch directory

set -e

OUTPUT_DIR="./experiments/transfer_study"
RESULTS_CSV="$OUTPUT_DIR/transfer_results_v6.csv"

echo "=============================================="
echo "Transfer Learning Evaluation (v6 - All Correct)"
echo "=============================================="

# Initialize results
echo "model,train_size,test_size,test_accuracy,test_loss" > "$RESULTS_CSV"

# Model configurations: model_name:input_type
declare -A MODEL_INPUT_TYPES=(
    ["diagonal_cnn"]="diagonal"
    ["scalable_diagonal"]="matrix"
    ["conv_dense"]="matrix"
    ["image_resnet"]="image"
)

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
    echo "=============================================="
    echo "Testing on: $TEST_DIR (n=$TEST_SIZE)"
    echo "=============================================="
    
    for MODEL in diagonal_cnn scalable_diagonal conv_dense image_resnet; do
        INPUT_TYPE="${MODEL_INPUT_TYPES[$MODEL]}"
        TRAIN_OUTPUT="$OUTPUT_DIR/trained_${MODEL}"
        MODEL_PATH=$(find "$TRAIN_OUTPUT" -name "best_model.pt" -type f 2>/dev/null | head -1)
        
        if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
            echo "  $MODEL: no trained model found"
            echo "$MODEL,128,$TEST_SIZE,N/A,N/A" >> "$RESULTS_CSV"
            continue
        fi
        
        echo ""
        echo "  Evaluating $MODEL (input_type=$INPUT_TYPE)..."
        
        python3 << PYEOF
import torch
import json
import sys
from pathlib import Path
sys.path.insert(0, '.')

MODEL_NAME = '$MODEL'
MODEL_PATH = '$MODEL_PATH'
TEST_DIR = '$TEST_DIR'
TEST_SIZE = '$TEST_SIZE'
INPUT_TYPE = '$INPUT_TYPE'

# Load config
config_path = Path(MODEL_PATH).parent / 'config.json'
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
else:
    config = {}

# Get parameters from config
train_n_size = config.get('n_size', config.get('matrix_size', 128))
num_classes = config.get('num_classes', 8)
task = config.get('task', 'classification')

print(f"  Config: n_size={train_n_size}, task={task}")

# Import model classes
from models import DiagonalCNN, ConvDenseNet, ImageResNet
try:
    from models import ScalableDiagonalCNN
except ImportError:
    ScalableDiagonalCNN = None

# Create model
try:
    if MODEL_NAME == 'diagonal_cnn':
        model = DiagonalCNN(n_size=train_n_size, num_classes=num_classes, task=task)
    elif MODEL_NAME == 'scalable_diagonal':
        if ScalableDiagonalCNN is None:
            raise ImportError("ScalableDiagonalCNN not found")
        model = ScalableDiagonalCNN(n_size=train_n_size, num_classes=num_classes, task=task)
    elif MODEL_NAME == 'conv_dense':
        model = ConvDenseNet(n_size=train_n_size, num_classes=num_classes, task=task)
    elif MODEL_NAME == 'image_resnet':
        model = ImageResNet(n_size=train_n_size, num_classes=num_classes, task=task)
    else:
        raise ValueError(f"Unknown model: {MODEL_NAME}")
except Exception as e:
    print(f"  Error creating model: {e}")
    print(f"RESULT,{MODEL_NAME},128,{TEST_SIZE},N/A,N/A")
    sys.exit(0)

# Load weights
try:
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
except Exception as e:
    print(f"  Error loading weights: {e}")
    print(f"RESULT,{MODEL_NAME},128,{TEST_SIZE},N/A,N/A")
    sys.exit(0)

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load test data with CORRECT input_type
from data import BlockJacobiDataset
from torch.utils.data import DataLoader

try:
    test_dataset = BlockJacobiDataset(
        data_dir=TEST_DIR,
        input_type=INPUT_TYPE,  # 'diagonal', 'matrix', or 'image'
        task=task
    )
    print(f"  Loaded {len(test_dataset)} samples (input_type={INPUT_TYPE})")
except ValueError as e:
    # Usually means needs .npz files that don't exist
    print(f"  Data error: {e}")
    print(f"RESULT,{MODEL_NAME},128,{TEST_SIZE},N/A,N/A")
    sys.exit(0)
except Exception as e:
    print(f"  Error loading data: {e}")
    print(f"RESULT,{MODEL_NAME},128,{TEST_SIZE},N/A,N/A")
    sys.exit(0)

if len(test_dataset) == 0:
    print("  No samples loaded!")
    print(f"RESULT,{MODEL_NAME},128,{TEST_SIZE},N/A,N/A")
    sys.exit(0)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Evaluate
correct = 0
total = 0
total_loss = 0.0
criterion = torch.nn.CrossEntropyLoss()

try:
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
except RuntimeError as e:
    if 'size mismatch' in str(e).lower() or 'shape' in str(e).lower():
        print(f"  SIZE MISMATCH: Cannot transfer to n={TEST_SIZE}")
    else:
        print(f"  Runtime error: {e}")
    print(f"RESULT,{MODEL_NAME},128,{TEST_SIZE},N/A,N/A")
    sys.exit(0)

if total > 0:
    accuracy = correct / total
    avg_loss = total_loss / total
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"RESULT,{MODEL_NAME},128,{TEST_SIZE},{accuracy:.4f},{avg_loss:.4f}")
else:
    print(f"RESULT,{MODEL_NAME},128,{TEST_SIZE},N/A,N/A")
PYEOF
        
    done
done | tee "$OUTPUT_DIR/transfer_eval_v6.log"

# Extract results
grep "^RESULT," "$OUTPUT_DIR/transfer_eval_v6.log" | sed 's/RESULT,//' >> "$RESULTS_CSV"

echo ""
echo "=============================================="
echo "FINAL RESULTS"
echo "=============================================="
cat "$RESULTS_CSV" | column -t -s','
echo ""
echo "Saved to: $RESULTS_CSV"
