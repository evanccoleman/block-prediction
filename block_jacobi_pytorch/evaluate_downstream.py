#!/usr/bin/env python3
"""
Downstream GMRES Performance Evaluation

Evaluates CNN predictions by comparing actual GMRES performance (iterations, time)
against baselines. This is the most compelling metric for reviewers.

Key comparison:
- CNN Prediction: Run GMRES with CNN-predicted block size
- Oracle: Best possible (from exhaustive profiling)
- Baselines: Fixed 10%, fixed 20%, no preconditioning

Usage:
    # After running suitesparse_pipeline.py to profile matrices:
    python evaluate_downstream.py \
        --eval-dir ./suitesparse_eval \
        --model-checkpoint ./checkpoints/image_resnet.pt \
        --output ./results/suitesparse_downstream.json

    # Or with simulated predictions (for testing the pipeline):
    python evaluate_downstream.py \
        --eval-dir ./suitesparse_eval \
        --simulate-predictions \
        --output ./results/test_downstream.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any
import numpy as np

# For loading images
from PIL import Image

FRACTION_CLASSES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


@dataclass
class MatrixResult:
    """Results for a single matrix."""
    matrix_id: str
    n: int
    nnz: int
    
    # Oracle (best possible)
    oracle_frac: float
    oracle_iters: int
    oracle_time: float
    
    # CNN prediction
    cnn_frac: Optional[float]
    cnn_iters: Optional[int]
    cnn_time: Optional[float]
    
    # Baselines
    fixed_10_iters: Optional[int]
    fixed_10_time: Optional[float]
    fixed_20_iters: Optional[int]
    fixed_20_time: Optional[float]
    
    # Classification accuracy (bonus)
    cnn_exact_match: bool = False
    cnn_within_1: bool = False
    
    # Performance ratios
    cnn_vs_oracle_iters: Optional[float] = None  # 1.0 = matches oracle
    cnn_vs_oracle_time: Optional[float] = None
    cnn_vs_fixed10_iters: Optional[float] = None  # <1.0 = CNN is better
    cnn_vs_fixed10_time: Optional[float] = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CNN predictions by downstream GMRES performance"
    )
    
    parser.add_argument("--eval-dir", "-e", type=str, required=True,
                        help="Directory with profiled matrices (from suitesparse_pipeline.py)")
    parser.add_argument("--output", "-o", type=str, default="./downstream_results.json",
                        help="Output JSON file for results")
    
    # Model options (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model-checkpoint", type=str,
                             help="Path to trained model checkpoint (.pt file)")
    model_group.add_argument("--predictions-file", type=str,
                             help="JSON file with pre-computed predictions {matrix_id: frac}")
    model_group.add_argument("--simulate-predictions", action="store_true",
                             help="Simulate predictions (for testing pipeline)")
    
    # Model config (if using checkpoint)
    parser.add_argument("--model-type", type=str, default="image_resnet",
                        choices=["image_resnet", "diagonal_cnn", "scalable_diagonal"],
                        help="Model architecture (default: image_resnet)")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Expected image size for model (default: 128)")
    
    # Output options
    parser.add_argument("--latex", action="store_true",
                        help="Also output LaTeX table")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-matrix results")
    
    return parser.parse_args()


def load_profiled_results(eval_dir: Path) -> Dict[str, Dict]:
    """Load all profiled matrix results from metadata directory."""
    meta_dir = eval_dir / "metadata"
    results = {}
    
    for meta_file in meta_dir.glob("*.json"):
        with open(meta_file) as f:
            data = json.load(f)
        matrix_id = data["matrix_id"]
        results[matrix_id] = data
    
    print(f"Loaded {len(results)} profiled matrices from {eval_dir}")
    return results


def get_performance_for_frac(perf_data: Dict, frac: float) -> tuple:
    """
    Get (iterations, total_time) for a given block fraction.
    Returns (None, None) if that fraction didn't converge.
    """
    # Handle both string and float keys
    key = str(frac)
    if key not in perf_data:
        key = f"{frac:.2f}"
    if key not in perf_data:
        # Try without leading zero
        key = f"{frac}"
    
    if key not in perf_data:
        return None, None
    
    entry = perf_data[key]
    if entry.get("status") != "converged":
        return None, None
    
    return entry.get("iterations"), entry.get("total_time")


def find_oracle(perf_data: Dict) -> tuple:
    """Find the oracle (best) block fraction by total_time."""
    best_frac = None
    best_time = float('inf')
    best_iters = None
    
    for frac_str, entry in perf_data.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "converged":
            continue
        
        time = entry.get("total_time", float('inf'))
        if time < best_time:
            best_time = time
            best_frac = float(frac_str)
            best_iters = entry.get("iterations")
    
    return best_frac, best_iters, best_time


class ModelPredictor:
    """Wrapper for CNN model prediction."""
    
    def __init__(self, checkpoint_path: str, model_type: str, image_size: int):
        self.image_size = image_size
        self.model_type = model_type
        self.model = self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path: str):
        """Load PyTorch model from checkpoint."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for model inference. pip install torch")
        
        # TODO: Import your actual model classes here
        # This is a placeholder - you'll need to adapt to your model code
        print(f"Loading model from {checkpoint_path}")
        print(f"Model type: {self.model_type}")
        
        # Placeholder: load state dict
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # You'll need to instantiate your actual model here
        # Example:
        # if self.model_type == "image_resnet":
        #     from your_models import ImageResNet
        #     model = ImageResNet(num_classes=8)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # model.eval()
        # return model
        
        raise NotImplementedError(
            "Model loading not implemented. Please adapt _load_model() "
            "to import and instantiate your model classes."
        )
    
    def predict(self, image_path: Path) -> float:
        """Predict block fraction for a matrix image."""
        import torch
        from torchvision import transforms
        
        # Load and preprocess image
        img = Image.open(image_path).convert('L')  # Grayscale
        
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
        img_tensor = transform(img).unsqueeze(0)  # Add batch dim
        
        with torch.no_grad():
            logits = self.model(img_tensor)
            pred_class = logits.argmax(dim=1).item()
        
        return FRACTION_CLASSES[pred_class]


class SimulatedPredictor:
    """Simulates predictions for pipeline testing."""
    
    def __init__(self, mode: str = "noisy_oracle"):
        """
        Modes:
        - "oracle": Always predict the optimal
        - "noisy_oracle": Oracle with noise (realistic accuracy ~80%)
        - "fixed": Always predict 0.15
        - "random": Random predictions
        """
        self.mode = mode
        self.rng = np.random.default_rng(42)
    
    def predict(self, matrix_data: Dict) -> float:
        """Predict based on matrix metadata (has access to oracle for simulation)."""
        oracle_frac = matrix_data["labels"].get("class_optimal_time")
        
        if self.mode == "oracle":
            return oracle_frac
        
        elif self.mode == "noisy_oracle":
            # 80% chance of correct, 15% off-by-one, 5% random
            r = self.rng.random()
            if r < 0.80:
                return oracle_frac
            elif r < 0.95:
                # Off by one class
                idx = FRACTION_CLASSES.index(oracle_frac)
                offset = self.rng.choice([-1, 1])
                new_idx = max(0, min(len(FRACTION_CLASSES)-1, idx + offset))
                return FRACTION_CLASSES[new_idx]
            else:
                return self.rng.choice(FRACTION_CLASSES)
        
        elif self.mode == "fixed":
            return 0.15
        
        elif self.mode == "random":
            return self.rng.choice(FRACTION_CLASSES)
        
        else:
            return oracle_frac


class FilePredictor:
    """Load predictions from a JSON file."""
    
    def __init__(self, predictions_file: str):
        with open(predictions_file) as f:
            data = json.load(f)
        
        # Handle both flat format {matrix_id: frac} and nested format {predictions: {...}}
        if "predictions" in data:
            self.predictions = data["predictions"]
            self.confidence_scores = data.get("confidence_scores", {})
            print(f"Loaded {len(self.predictions)} predictions from {predictions_file}")
            print(f"  Checkpoint: {data.get('checkpoint', 'unknown')}")
            print(f"  Training n_size: {data.get('n_size', 'unknown')}")
        else:
            self.predictions = data
            self.confidence_scores = {}
            print(f"Loaded {len(self.predictions)} predictions from {predictions_file}")
    
    def predict(self, matrix_id: str) -> Optional[float]:
        return self.predictions.get(matrix_id)


def evaluate_matrix(matrix_data: Dict, predictor, eval_dir: Path) -> Optional[MatrixResult]:
    """Evaluate a single matrix."""
    matrix_id = matrix_data["matrix_id"]
    props = matrix_data["matrix_properties"]
    perf_data = matrix_data["performance_data"]
    
    # Find oracle
    oracle_frac, oracle_iters, oracle_time = find_oracle(perf_data)
    if oracle_frac is None:
        print(f"  Skipping {matrix_id}: no converged block sizes")
        return None
    
    # Get CNN prediction
    if isinstance(predictor, SimulatedPredictor):
        cnn_frac = predictor.predict(matrix_data)
    elif isinstance(predictor, FilePredictor):
        cnn_frac = predictor.predict(matrix_id)
    else:
        # ModelPredictor - needs image path
        image_path = eval_dir / "images" / matrix_data["files"]["image"]
        cnn_frac = predictor.predict(image_path)
    
    if cnn_frac is None:
        print(f"  Skipping {matrix_id}: no prediction available")
        return None
    
    # Get performance for CNN prediction
    cnn_iters, cnn_time = get_performance_for_frac(perf_data, cnn_frac)
    
    # Get baseline performances
    fixed_10_iters, fixed_10_time = get_performance_for_frac(perf_data, 0.10)
    fixed_20_iters, fixed_20_time = get_performance_for_frac(perf_data, 0.20)
    
    # Classification accuracy
    exact_match = (cnn_frac == oracle_frac)
    oracle_idx = FRACTION_CLASSES.index(oracle_frac) if oracle_frac in FRACTION_CLASSES else -1
    cnn_idx = FRACTION_CLASSES.index(cnn_frac) if cnn_frac in FRACTION_CLASSES else -1
    within_1 = abs(oracle_idx - cnn_idx) <= 1 if oracle_idx >= 0 and cnn_idx >= 0 else False
    
    result = MatrixResult(
        matrix_id=matrix_id,
        n=props["size"],
        nnz=props["nnz"],
        oracle_frac=oracle_frac,
        oracle_iters=oracle_iters,
        oracle_time=oracle_time,
        cnn_frac=cnn_frac,
        cnn_iters=cnn_iters,
        cnn_time=cnn_time,
        fixed_10_iters=fixed_10_iters,
        fixed_10_time=fixed_10_time,
        fixed_20_iters=fixed_20_iters,
        fixed_20_time=fixed_20_time,
        cnn_exact_match=exact_match,
        cnn_within_1=within_1,
    )
    
    # Compute ratios (if data available)
    if cnn_iters and oracle_iters:
        result.cnn_vs_oracle_iters = cnn_iters / oracle_iters
    if cnn_time and oracle_time:
        result.cnn_vs_oracle_time = cnn_time / oracle_time
    if cnn_iters and fixed_10_iters:
        result.cnn_vs_fixed10_iters = cnn_iters / fixed_10_iters
    if cnn_time and fixed_10_time:
        result.cnn_vs_fixed10_time = cnn_time / fixed_10_time
    
    return result


def compute_aggregate_stats(results: List[MatrixResult]) -> Dict:
    """Compute aggregate statistics across all matrices."""
    n = len(results)
    
    # Filter results with valid CNN predictions
    valid = [r for r in results if r.cnn_iters is not None]
    n_valid = len(valid)
    
    # Classification accuracy
    exact_matches = sum(1 for r in valid if r.cnn_exact_match)
    within_1_matches = sum(1 for r in valid if r.cnn_within_1)
    
    # Performance ratios (vs oracle)
    oracle_ratios_iters = [r.cnn_vs_oracle_iters for r in valid if r.cnn_vs_oracle_iters]
    oracle_ratios_time = [r.cnn_vs_oracle_time for r in valid if r.cnn_vs_oracle_time]
    
    # Performance ratios (vs fixed 10%)
    fixed10_ratios_iters = [r.cnn_vs_fixed10_iters for r in valid if r.cnn_vs_fixed10_iters]
    fixed10_ratios_time = [r.cnn_vs_fixed10_time for r in valid if r.cnn_vs_fixed10_time]
    
    # Iteration counts
    cnn_iters = [r.cnn_iters for r in valid if r.cnn_iters]
    oracle_iters = [r.oracle_iters for r in valid if r.oracle_iters]
    fixed10_iters = [r.fixed_10_iters for r in valid if r.fixed_10_iters]
    
    def safe_stats(arr):
        if not arr:
            return {"mean": None, "median": None, "std": None, "min": None, "max": None}
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    
    return {
        "n_matrices": n,
        "n_valid_predictions": n_valid,
        
        "classification": {
            "exact_accuracy": exact_matches / n_valid if n_valid else 0,
            "within_1_accuracy": within_1_matches / n_valid if n_valid else 0,
            "exact_matches": exact_matches,
            "within_1_matches": within_1_matches,
        },
        
        "cnn_vs_oracle": {
            "iterations_ratio": safe_stats(oracle_ratios_iters),
            "time_ratio": safe_stats(oracle_ratios_time),
            "pct_matching_oracle": sum(1 for r in oracle_ratios_iters if r == 1.0) / len(oracle_ratios_iters) if oracle_ratios_iters else 0,
            "pct_within_10pct": sum(1 for r in oracle_ratios_iters if r <= 1.1) / len(oracle_ratios_iters) if oracle_ratios_iters else 0,
        },
        
        "cnn_vs_fixed10": {
            "iterations_ratio": safe_stats(fixed10_ratios_iters),
            "time_ratio": safe_stats(fixed10_ratios_time),
            "pct_cnn_better": sum(1 for r in fixed10_ratios_iters if r < 1.0) / len(fixed10_ratios_iters) if fixed10_ratios_iters else 0,
            "pct_cnn_much_better": sum(1 for r in fixed10_ratios_iters if r < 0.8) / len(fixed10_ratios_iters) if fixed10_ratios_iters else 0,
        },
        
        "absolute_iterations": {
            "cnn": safe_stats(cnn_iters),
            "oracle": safe_stats(oracle_iters),
            "fixed_10": safe_stats(fixed10_iters),
        },
    }


def generate_latex_table(results: List[MatrixResult], stats: Dict) -> str:
    """Generate LaTeX table for paper."""
    
    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{Downstream GMRES performance on SuiteSparse matrices.}",
        r"    \label{tab:suitesparse-downstream}",
        r"    \begin{tabular}{lcccc}",
        r"        \toprule",
        r"        \textbf{Method} & \textbf{Mean Iters} & \textbf{Median Iters} & \textbf{vs Oracle} & \textbf{vs Fixed-10\%} \\",
        r"        \midrule",
    ]
    
    abs_stats = stats["absolute_iterations"]
    
    # Oracle row
    oracle_mean = abs_stats["oracle"]["mean"]
    oracle_median = abs_stats["oracle"]["median"]
    lines.append(f"        Oracle & {oracle_mean:.1f} & {oracle_median:.1f} & 1.00$\\times$ & --- \\\\")
    
    # CNN row
    cnn_mean = abs_stats["cnn"]["mean"]
    cnn_median = abs_stats["cnn"]["median"]
    cnn_vs_oracle = stats["cnn_vs_oracle"]["iterations_ratio"]["mean"]
    cnn_vs_fixed = stats["cnn_vs_fixed10"]["iterations_ratio"]["mean"]
    lines.append(f"        CNN (ours) & {cnn_mean:.1f} & {cnn_median:.1f} & {cnn_vs_oracle:.2f}$\\times$ & {cnn_vs_fixed:.2f}$\\times$ \\\\")
    
    # Fixed 10% row
    f10_mean = abs_stats["fixed_10"]["mean"]
    f10_median = abs_stats["fixed_10"]["median"]
    if f10_mean and oracle_mean:
        f10_vs_oracle = f10_mean / oracle_mean
        lines.append(f"        Fixed 10\\% & {f10_mean:.1f} & {f10_median:.1f} & {f10_vs_oracle:.2f}$\\times$ & 1.00$\\times$ \\\\")
    
    lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def main():
    args = parse_args()
    
    eval_dir = Path(args.eval_dir)
    
    # Load profiled results
    profiled = load_profiled_results(eval_dir)
    
    if not profiled:
        print("No profiled matrices found!")
        return
    
    # Set up predictor
    if args.simulate_predictions:
        print("Using simulated predictions (noisy oracle, ~80% accuracy)")
        predictor = SimulatedPredictor(mode="noisy_oracle")
    elif args.predictions_file:
        predictor = FilePredictor(args.predictions_file)
    else:
        predictor = ModelPredictor(
            args.model_checkpoint,
            args.model_type,
            args.image_size
        )
    
    # Evaluate each matrix
    results = []
    for matrix_id, matrix_data in profiled.items():
        if args.verbose:
            print(f"Evaluating {matrix_id}...")
        
        result = evaluate_matrix(matrix_data, predictor, eval_dir)
        if result:
            results.append(result)
            
            if args.verbose:
                print(f"  Oracle: {result.oracle_frac:.2f} ({result.oracle_iters} iters)")
                print(f"  CNN:    {result.cnn_frac:.2f} ({result.cnn_iters} iters) "
                      f"[{'✓' if result.cnn_exact_match else '✗'}]")
                if result.cnn_vs_oracle_iters:
                    print(f"  Ratio:  {result.cnn_vs_oracle_iters:.2f}x oracle")
    
    print(f"\nEvaluated {len(results)} matrices")
    
    # Compute aggregate statistics
    stats = compute_aggregate_stats(results)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\nClassification Accuracy:")
    print(f"  Exact match:  {stats['classification']['exact_accuracy']:.1%}")
    print(f"  Within-1:     {stats['classification']['within_1_accuracy']:.1%}")
    
    print(f"\nCNN vs Oracle (iteration ratio, lower is better):")
    print(f"  Mean:   {stats['cnn_vs_oracle']['iterations_ratio']['mean']:.2f}x")
    print(f"  Median: {stats['cnn_vs_oracle']['iterations_ratio']['median']:.2f}x")
    print(f"  % matching oracle: {stats['cnn_vs_oracle']['pct_matching_oracle']:.1%}")
    print(f"  % within 10%:      {stats['cnn_vs_oracle']['pct_within_10pct']:.1%}")
    
    print(f"\nCNN vs Fixed-10% baseline (ratio, <1 means CNN is better):")
    print(f"  Mean:   {stats['cnn_vs_fixed10']['iterations_ratio']['mean']:.2f}x")
    print(f"  % CNN better:      {stats['cnn_vs_fixed10']['pct_cnn_better']:.1%}")
    print(f"  % CNN much better: {stats['cnn_vs_fixed10']['pct_cnn_much_better']:.1%}")
    
    print(f"\nAbsolute Iterations:")
    print(f"  Oracle mean:    {stats['absolute_iterations']['oracle']['mean']:.1f}")
    print(f"  CNN mean:       {stats['absolute_iterations']['cnn']['mean']:.1f}")
    print(f"  Fixed-10% mean: {stats['absolute_iterations']['fixed_10']['mean']:.1f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "summary": stats,
        "per_matrix": [asdict(r) for r in results],
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # LaTeX table
    if args.latex:
        latex = generate_latex_table(results, stats)
        latex_path = output_path.with_suffix('.tex')
        with open(latex_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to: {latex_path}")
        print("\nLaTeX preview:")
        print(latex)


if __name__ == "__main__":
    main()
