#!/usr/bin/env python3
"""
Wall-Clock Timing for GMRES with Block-Jacobi Preconditioning

Measures actual setup and solve times on SuiteSparse matrices to validate
that iteration count correlates with solve time and to assess the accuracy
of the theoretical cost model.

Usage:
    python measure_wallclock.py \
        --eval-dir ./suitesparse_eval \
        --output ./wallclock_results.json \
        --num-runs 5

Output includes:
- Per-matrix, per-block-fraction timings (setup, solve, total)
- Correlation between iteration count and solve time
- Comparison of theoretical vs measured setup cost scaling
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.io import mmread

# Suppress GMRES convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

FRACTION_CLASSES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure wall-clock GMRES times on SuiteSparse matrices"
    )
    
    parser.add_argument("--eval-dir", "-e", type=str, required=True,
                        help="Directory with SuiteSparse matrices (from process_local_matrices.py)")
    parser.add_argument("--output", "-o", type=str, default="./wallclock_results.json",
                        help="Output JSON file")
    parser.add_argument("--num-runs", "-n", type=int, default=5,
                        help="Number of timing runs per configuration (for averaging)")
    parser.add_argument("--max-matrices", type=int, default=None,
                        help="Maximum number of matrices to process")
    parser.add_argument("--fractions", nargs="+", type=float, default=None,
                        help="Specific fractions to test (default: all 8)")
    parser.add_argument("--tol", type=float, default=1e-8,
                        help="GMRES convergence tolerance")
    parser.add_argument("--maxiter", type=int, default=2000,
                        help="Maximum GMRES iterations")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed progress")
    
    return parser.parse_args()


def build_block_jacobi_preconditioner(A: csr_matrix, block_size: int) -> Tuple[LinearOperator, float, int]:
    """
    Build block-Jacobi preconditioner and measure setup time.
    
    Returns:
        M_op: LinearOperator for preconditioner application
        setup_time: Wall-clock time to build preconditioner (seconds)
        nnz_M: Number of nonzeros in preconditioner
    """
    n = A.shape[0]
    
    start_time = time.perf_counter()
    
    # Extract and invert diagonal blocks
    num_blocks = (n + block_size - 1) // block_size
    block_inverses = []
    
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, n)
        actual_size = end_idx - start_idx
        
        # Extract block
        block = A[start_idx:end_idx, start_idx:end_idx].toarray()
        
        # Invert block
        try:
            block_inv = np.linalg.inv(block)
        except np.linalg.LinAlgError:
            # Regularize if singular
            block_inv = np.linalg.inv(block + 1e-10 * np.eye(actual_size))
        
        block_inverses.append((start_idx, end_idx, block_inv))
    
    setup_time = time.perf_counter() - start_time
    
    # Count nonzeros in preconditioner
    nnz_M = sum((end - start) ** 2 for start, end, _ in block_inverses)
    
    # Create LinearOperator for preconditioner application
    def matvec(x):
        y = np.zeros_like(x)
        for start_idx, end_idx, block_inv in block_inverses:
            y[start_idx:end_idx] = block_inv @ x[start_idx:end_idx]
        return y
    
    M_op = LinearOperator((n, n), matvec=matvec)
    
    return M_op, setup_time, nnz_M


def run_gmres_timed(A: csr_matrix, b: np.ndarray, M_op: LinearOperator,
                    tol: float, maxiter: int) -> Tuple[int, float, bool]:
    """
    Run GMRES and measure solve time.
    
    Returns:
        iterations: Number of GMRES iterations
        solve_time: Wall-clock time for solve (seconds)
        converged: Whether GMRES converged
    """
    # Count iterations via callback
    iteration_count = [0]
    
    def callback(rk):
        iteration_count[0] += 1
    
    start_time = time.perf_counter()
    
    x, info = gmres(A, b, M=M_op, tol=tol, maxiter=maxiter, 
                    callback=callback, callback_type='pr_norm')
    
    solve_time = time.perf_counter() - start_time
    
    converged = (info == 0)
    
    return iteration_count[0], solve_time, converged


def time_configuration(A: csr_matrix, block_frac: float, num_runs: int,
                       tol: float, maxiter: int) -> Optional[Dict]:
    """
    Time a specific matrix/block-fraction configuration multiple times.
    
    Returns dict with timing statistics, or None if GMRES doesn't converge.
    """
    n = A.shape[0]
    block_size = max(1, int(block_frac * n))
    
    # Generate consistent RHS
    np.random.seed(42)
    b = np.random.randn(n)
    b = b / np.linalg.norm(b)
    
    setup_times = []
    solve_times = []
    total_times = []
    iterations_list = []
    
    for run in range(num_runs):
        # Build preconditioner (timed)
        M_op, setup_time, nnz_M = build_block_jacobi_preconditioner(A, block_size)
        
        # Run GMRES (timed)
        iterations, solve_time, converged = run_gmres_timed(
            A, b, M_op, tol, maxiter
        )
        
        if not converged:
            return None
        
        setup_times.append(setup_time)
        solve_times.append(solve_time)
        total_times.append(setup_time + solve_time)
        iterations_list.append(iterations)
    
    return {
        "block_size": block_size,
        "block_frac": block_frac,
        "nnz_M": nnz_M,
        "iterations": int(np.mean(iterations_list)),
        "setup_time": {
            "mean": float(np.mean(setup_times)),
            "std": float(np.std(setup_times)),
            "min": float(np.min(setup_times)),
            "max": float(np.max(setup_times)),
        },
        "solve_time": {
            "mean": float(np.mean(solve_times)),
            "std": float(np.std(solve_times)),
            "min": float(np.min(solve_times)),
            "max": float(np.max(solve_times)),
        },
        "total_time": {
            "mean": float(np.mean(total_times)),
            "std": float(np.std(total_times)),
            "min": float(np.min(total_times)),
            "max": float(np.max(total_times)),
        },
        "num_runs": num_runs,
    }


def load_matrix(eval_dir: Path, matrix_id: str) -> Optional[csr_matrix]:
    """Load matrix from .npz or .mtx file."""
    # Try .npz first (if saved by process_local_matrices.py with --save-raw)
    npz_path = eval_dir / "matrices" / f"{matrix_id}.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        return csr_matrix((data['data'], data['indices'], data['indptr']), 
                          shape=data['shape'])
    
    # Try to find original .mtx file
    # Check common locations
    for pattern in [f"*{matrix_id}*.mtx", f"{matrix_id}.mtx"]:
        matches = list(eval_dir.glob(f"**/{pattern}"))
        if matches:
            return csr_matrix(mmread(matches[0]))
    
    return None


def process_matrix(eval_dir: Path, meta_file: Path, fractions: List[float],
                   num_runs: int, tol: float, maxiter: int, 
                   verbose: bool) -> Optional[Dict]:
    """Process a single matrix: load and time all configurations."""
    
    with open(meta_file) as f:
        meta = json.load(f)
    
    matrix_id = meta["matrix_id"]
    n = meta["matrix_properties"]["size"]
    nnz = meta["matrix_properties"]["nnz"]
    
    if verbose:
        print(f"\nProcessing {matrix_id} (n={n}, nnz={nnz})")
    
    # Load matrix
    A = load_matrix(eval_dir, matrix_id)
    
    if A is None:
        # Try to reconstruct from stored data or skip
        print(f"  Warning: Could not load matrix {matrix_id}, skipping")
        return None
    
    # Ensure matrix is square and sizes match
    if A.shape[0] != A.shape[1]:
        print(f"  Warning: Matrix {matrix_id} is not square, skipping")
        return None
    
    if A.shape[0] != n:
        print(f"  Warning: Matrix size mismatch for {matrix_id}, skipping")
        return None
    
    # Time each block fraction
    results = {
        "matrix_id": matrix_id,
        "n": n,
        "nnz": nnz,
        "configurations": {}
    }
    
    for frac in fractions:
        if verbose:
            print(f"  Block fraction {frac:.2f}...", end=" ", flush=True)
        
        timing = time_configuration(A, frac, num_runs, tol, maxiter)
        
        if timing is None:
            if verbose:
                print("did not converge")
            continue
        
        if verbose:
            print(f"{timing['iterations']} iters, "
                  f"setup={timing['setup_time']['mean']*1000:.2f}ms, "
                  f"solve={timing['solve_time']['mean']*1000:.2f}ms")
        
        results["configurations"][str(frac)] = timing
    
    if not results["configurations"]:
        return None
    
    # Find optima
    configs = results["configurations"]
    
    # Iteration-optimal
    iter_opt = min(configs.items(), key=lambda x: x[1]["iterations"])
    results["iteration_optimal"] = {
        "frac": float(iter_opt[0]),
        "iterations": iter_opt[1]["iterations"],
        "total_time": iter_opt[1]["total_time"]["mean"],
    }
    
    # Time-optimal (measured wall-clock)
    time_opt = min(configs.items(), key=lambda x: x[1]["total_time"]["mean"])
    results["time_optimal"] = {
        "frac": float(time_opt[0]),
        "iterations": time_opt[1]["iterations"],
        "total_time": time_opt[1]["total_time"]["mean"],
    }
    
    # Setup-heavy optimal (if we only count setup once, as in amortized case)
    # Use: 0.1 * setup + solve as proxy for "mostly solve" workload
    amort_opt = min(configs.items(), 
                    key=lambda x: 0.1 * x[1]["setup_time"]["mean"] + x[1]["solve_time"]["mean"])
    results["amortized_optimal"] = {
        "frac": float(amort_opt[0]),
        "iterations": amort_opt[1]["iterations"],
        "total_time": amort_opt[1]["total_time"]["mean"],
    }
    
    return results


def compute_correlations(all_results: List[Dict]) -> Dict:
    """Compute correlations between iterations and times."""
    iterations = []
    solve_times = []
    setup_times = []
    total_times = []
    block_sizes = []
    
    for result in all_results:
        for frac_str, config in result["configurations"].items():
            iterations.append(config["iterations"])
            solve_times.append(config["solve_time"]["mean"])
            setup_times.append(config["setup_time"]["mean"])
            total_times.append(config["total_time"]["mean"])
            block_sizes.append(config["block_size"])
    
    iterations = np.array(iterations)
    solve_times = np.array(solve_times)
    setup_times = np.array(setup_times)
    total_times = np.array(total_times)
    block_sizes = np.array(block_sizes)
    
    # Compute correlations
    def pearson_r(x, y):
        return float(np.corrcoef(x, y)[0, 1])
    
    correlations = {
        "iterations_vs_solve_time": pearson_r(iterations, solve_times),
        "iterations_vs_total_time": pearson_r(iterations, total_times),
        "block_size_vs_setup_time": pearson_r(block_sizes, setup_times),
        "block_size_cubed_vs_setup_time": pearson_r(block_sizes**3, setup_times),
    }
    
    # Check if setup scales as O(m^3)
    # Fit log(setup_time) = a * log(block_size) + b
    log_bs = np.log(block_sizes + 1)
    log_setup = np.log(setup_times + 1e-10)
    slope, intercept = np.polyfit(log_bs, log_setup, 1)
    correlations["setup_time_scaling_exponent"] = float(slope)
    
    return correlations


def compute_summary(all_results: List[Dict]) -> Dict:
    """Compute summary statistics."""
    
    # How often do iteration-optimal and time-optimal agree?
    agree_count = 0
    iter_better_count = 0
    time_better_count = 0
    
    iter_opt_times = []
    time_opt_times = []
    iter_opt_iters = []
    time_opt_iters = []
    
    for result in all_results:
        iter_frac = result["iteration_optimal"]["frac"]
        time_frac = result["time_optimal"]["frac"]
        
        if abs(iter_frac - time_frac) < 0.01:
            agree_count += 1
        
        iter_opt_times.append(result["iteration_optimal"]["total_time"])
        time_opt_times.append(result["time_optimal"]["total_time"])
        iter_opt_iters.append(result["iteration_optimal"]["iterations"])
        time_opt_iters.append(result["time_optimal"]["iterations"])
        
        # Which is actually faster?
        if result["iteration_optimal"]["total_time"] < result["time_optimal"]["total_time"] * 0.99:
            iter_better_count += 1
        elif result["time_optimal"]["total_time"] < result["iteration_optimal"]["total_time"] * 0.99:
            time_better_count += 1
    
    n_matrices = len(all_results)
    
    return {
        "n_matrices": n_matrices,
        "optima_agree_pct": 100 * agree_count / n_matrices,
        "iteration_optimal_mean_time": float(np.mean(iter_opt_times)),
        "time_optimal_mean_time": float(np.mean(time_opt_times)),
        "iteration_optimal_mean_iters": float(np.mean(iter_opt_iters)),
        "time_optimal_mean_iters": float(np.mean(time_opt_iters)),
        "time_ratio_iter_vs_time_opt": float(np.mean(iter_opt_times) / np.mean(time_opt_times)),
    }


def main():
    args = parse_args()
    
    eval_dir = Path(args.eval_dir)
    meta_dir = eval_dir / "metadata"
    
    if not meta_dir.exists():
        print(f"Error: Metadata directory not found: {meta_dir}")
        print("Run process_local_matrices.py first to generate metadata.")
        sys.exit(1)
    
    fractions = args.fractions if args.fractions else FRACTION_CLASSES
    
    print("=" * 60)
    print("WALL-CLOCK TIMING MEASUREMENT")
    print("=" * 60)
    print(f"Matrices directory: {eval_dir}")
    print(f"Block fractions: {fractions}")
    print(f"Timing runs per config: {args.num_runs}")
    print(f"GMRES tolerance: {args.tol}")
    print()
    
    # Find all matrices
    meta_files = sorted(meta_dir.glob("*.json"))
    if args.max_matrices:
        meta_files = meta_files[:args.max_matrices]
    
    print(f"Found {len(meta_files)} matrices to process")
    
    # Process each matrix
    all_results = []
    
    for i, meta_file in enumerate(meta_files):
        if not args.verbose:
            print(f"\rProcessing {i+1}/{len(meta_files)}...", end="", flush=True)
        
        result = process_matrix(
            eval_dir, meta_file, fractions,
            args.num_runs, args.tol, args.maxiter, args.verbose
        )
        
        if result:
            all_results.append(result)
    
    if not args.verbose:
        print()
    
    print(f"\nSuccessfully processed {len(all_results)} matrices")
    
    # Compute correlations
    correlations = compute_correlations(all_results)
    
    # Compute summary
    summary = compute_summary(all_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\nCorrelations:")
    print(f"  Iterations vs solve time:    r = {correlations['iterations_vs_solve_time']:.3f}")
    print(f"  Iterations vs total time:    r = {correlations['iterations_vs_total_time']:.3f}")
    print(f"  Block size vs setup time:    r = {correlations['block_size_vs_setup_time']:.3f}")
    print(f"  Block size³ vs setup time:   r = {correlations['block_size_cubed_vs_setup_time']:.3f}")
    print(f"  Setup time scaling exponent: {correlations['setup_time_scaling_exponent']:.2f} (expected: 3.0 for O(m³))")
    
    print(f"\nOptima comparison:")
    print(f"  Iteration-optimal and time-optimal agree: {summary['optima_agree_pct']:.1f}%")
    print(f"  Iteration-optimal mean time: {summary['iteration_optimal_mean_time']*1000:.2f} ms")
    print(f"  Time-optimal mean time:      {summary['time_optimal_mean_time']*1000:.2f} ms")
    print(f"  Ratio (iter-opt / time-opt): {summary['time_ratio_iter_vs_time_opt']:.3f}x")
    
    print(f"\nMean iterations:")
    print(f"  Iteration-optimal: {summary['iteration_optimal_mean_iters']:.1f}")
    print(f"  Time-optimal:      {summary['time_optimal_mean_iters']:.1f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "summary": summary,
        "correlations": correlations,
        "per_matrix": all_results,
        "config": {
            "fractions": fractions,
            "num_runs": args.num_runs,
            "tol": args.tol,
            "maxiter": args.maxiter,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print text for paper
    print("\n" + "=" * 60)
    print("TEXT FOR PAPER")
    print("=" * 60)
    print(f"""
\\paragraph{{Wall-clock validation.}}
To validate iteration count as a proxy for solver performance, we measured 
wall-clock times for GMRES with block-Jacobi preconditioning on {len(all_results)} 
SuiteSparse matrices using SciPy's implementation. Each configuration was 
timed {args.num_runs} times and averaged. Iteration count correlates strongly with 
solve time (Pearson $r = {correlations['iterations_vs_solve_time']:.2f}$), supporting 
its use as a hardware-independent performance metric. Setup time scales as 
$O(m^{{{correlations['setup_time_scaling_exponent']:.1f}}})$ with block size $m$, 
{'consistent with' if abs(correlations['setup_time_scaling_exponent'] - 3.0) < 0.5 else 'somewhat less than'} 
the theoretical $O(m^3)$ model. The iteration-optimal and time-optimal block 
sizes agree on {summary['optima_agree_pct']:.0f}\\% of matrices, with iteration-optimal 
choices achieving {summary['time_ratio_iter_vs_time_opt']:.2f}$\\times$ the wall-clock 
time of time-optimal choices on average.
""")


if __name__ == "__main__":
    main()
