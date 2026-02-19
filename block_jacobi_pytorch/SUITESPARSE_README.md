# SuiteSparse Evaluation Pipeline

Scripts for evaluating your CNN block-Jacobi models on real-world matrices from the SuiteSparse Matrix Collection.

## Quick Start

### Option 1: Automatic download with ssgetpy

```bash
# Install dependency
pip install ssgetpy

# Download and profile up to 50 SPD matrices with n <= 500
python suitesparse_pipeline.py --output ./suitesparse_eval --max-n 500 --max-matrices 50

# Just list what would be downloaded (dry run)
python suitesparse_pipeline.py --list-only --max-n 500
```

### Option 2: Manual download + local processing

If ssgetpy can't connect (firewall, etc.), manually download matrices:

1. Go to https://sparse.tamu.edu/
2. Filter: Rows ≤ 1000, Cols ≤ 1000, SPD = yes
3. Download .mtx files to a local folder

Then process them:

```bash
python process_local_matrices.py --input ./downloaded_matrices --output ./local_eval
```

## Output Format

Both scripts produce output compatible with your existing training pipeline:

```
suitesparse_eval/
├── images/
│   ├── matrix_HB_bcsstk01.png
│   ├── matrix_HB_bcsstk02.png
│   └── ...
├── metadata/
│   ├── matrix_HB_bcsstk01.json
│   ├── matrix_HB_bcsstk02.json
│   └── ...
├── matrices/           # Only if --save-raw
│   └── *.npz
└── summary.json
```

### Metadata JSON format

Each matrix gets a JSON file with:

```json
{
  "matrix_id": "HB_bcsstk01",
  "suitesparse_info": {
    "group": "HB",
    "name": "bcsstk01",
    "id": 3,
    "is_spd": true
  },
  "labels": {
    "class_optimal_time": 0.15,
    "class_optimal_iterations": 0.15,
    "regression_interpolated_optimal": 0.147
  },
  "matrix_properties": {
    "size": 48,
    "nnz": 400,
    "density": 0.174
  },
  "performance_data": {
    "0.05": {"status": "converged", "iterations": 12, ...},
    "0.10": {"status": "converged", "iterations": 8, ...},
    ...
  }
}
```

## Recommended Workflow

### Step 1: Get matrices

For a paper, I'd recommend:

```bash
# SPD matrices, focus on structural engineering (known block structure)
python suitesparse_pipeline.py \
    --output ./suitesparse_eval \
    --max-n 500 \
    --groups HB Boeing DNVS \
    --max-matrices 30

# Then expand to more general SPD
python suitesparse_pipeline.py \
    --output ./suitesparse_eval_general \
    --max-n 500 \
    --max-matrices 50 \
    --skip-existing
```

### Step 2: Run your trained models

```python
# evaluate_on_suitesparse.py (you'll create this)
import json
from pathlib import Path
from your_model import load_model, predict

# Load your trained model
model = load_model("checkpoints/image_resnet_n128.pt")

results = []
eval_dir = Path("suitesparse_eval")

for meta_file in (eval_dir / "metadata").glob("*.json"):
    with open(meta_file) as f:
        meta = json.load(meta_file)
    
    # Load image
    img_path = eval_dir / "images" / meta["files"]["image"]
    
    # Get prediction
    predicted_frac = predict(model, img_path)
    
    # Compare to ground truth
    true_frac = meta["labels"]["class_optimal_time"]
    
    results.append({
        "matrix_id": meta["matrix_id"],
        "n": meta["matrix_properties"]["size"],
        "predicted": predicted_frac,
        "actual": true_frac,
        "correct": predicted_frac == true_frac,
        "within_1": abs(FRACTION_CLASSES.index(predicted_frac) - 
                        FRACTION_CLASSES.index(true_frac)) <= 1
    })

# Report accuracy
accuracy = sum(r["correct"] for r in results) / len(results)
within_1 = sum(r["within_1"] for r in results) / len(results)
print(f"Exact accuracy: {accuracy:.1%}")
print(f"Within-1 accuracy: {within_1:.1%}")
```

### Step 3: Generate paper table

```python
import pandas as pd

df = pd.DataFrame(results)

# Summary by matrix size bucket
df["size_bucket"] = pd.cut(df["n"], bins=[0, 100, 250, 500, 1000], 
                           labels=["<100", "100-250", "250-500", "500-1000"])

summary = df.groupby("size_bucket").agg({
    "correct": "mean",
    "within_1": "mean",
    "matrix_id": "count"
}).rename(columns={"correct": "Accuracy", "within_1": "Within-1", "matrix_id": "Count"})

print(summary.to_latex())
```

## Useful Matrix Groups

For block-Jacobi evaluation, these groups have known block structure:

| Group | Description | Why useful |
|-------|-------------|------------|
| **HB** (Harwell-Boeing) | Classic test matrices | Well-documented, small |
| **BCSSTK** | Boeing structural stiffness | Natural 3×3 or 6×6 blocks (3D FEM DOFs) |
| **BCSSTM** | Boeing structural mass | Same structure as BCSSTK |
| **Boeing** | Aerospace structural | Block structure from physics |
| **DNVS** | Ship structures | 3D element blocks |
| **Cylshell** | Cylindrical shell FEM | Regular block structure |

## Command Reference

### suitesparse_pipeline.py

```
--output DIR          Output directory (required unless --list-only)
--cache-dir DIR       Cache for downloaded matrices (default: ./suitesparse_cache)
--max-n N             Maximum matrix dimension (default: 1000)
--min-n N             Minimum matrix dimension (default: 10)
--max-matrices N      Process at most N matrices
--include-non-spd     Include non-SPD matrices
--groups G1 G2 ...    Only these matrix groups
--image-size N        Output image resolution (default: 128)
--save-raw            Also save .npz matrix files
--list-only           Just list matrices, don't download
--skip-existing       Skip already-processed matrices
```

### process_local_matrices.py

```
--input DIR           Directory with .mtx files
--files F1 F2 ...     Specific .mtx files to process
--output DIR          Output directory (required)
--max-n N             Skip matrices larger than N
--image-size N        Output image resolution (default: 128)
--save-raw            Also save .npz matrix files
--skip-existing       Skip already-processed matrices
```

## Troubleshooting

### ssgetpy connection fails

If you see proxy/firewall errors:
1. Use `--list-only` to see what you need
2. Manually download from https://sparse.tamu.edu/
3. Use `process_local_matrices.py` instead

### Matrix fails to converge

Some matrices won't converge with any block size (ill-conditioned, etc.). 
The scripts record these as failed and continue. Check `summary.json` for stats.

### Out of memory

For larger matrices:
- Reduce `--max-n` 
- The scripts use sparse formats, but preconditioner inversion creates dense blocks
- Block size 40% of n=1000 means 400×400 dense blocks to invert

## Integration with Training Pipeline

The output format matches `png_builder2.py`. To use these matrices for evaluation:

```python
# In your evaluation code, just point to the suitesparse directory
eval_dataset = ImageClassificationDataset(
    image_dir="suitesparse_eval/images",
    metadata_dir="suitesparse_eval/metadata",
    # ... same args as synthetic data
)
```
