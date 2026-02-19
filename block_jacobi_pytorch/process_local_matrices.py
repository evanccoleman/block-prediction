#!/usr/bin/env python3
"""
Process Local Matrices

For matrices you've already downloaded (Matrix Market .mtx files),
this script profiles them with GMRES + block-Jacobi and saves results
in the same format as png_builder2.py.

Usage:
    # Process all .mtx files in a directory
    python process_local_matrices.py --input ./downloaded_matrices --output ./local_eval

    # Process specific files
    python process_local_matrices.py --files matrix1.mtx matrix2.mtx --output ./local_eval

This is useful when:
- Network restrictions prevent ssgetpy from working
- You want to test on specific matrices
- You've manually downloaded matrices from sparse.tamu.edu
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import glob

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres
from scipy.io import mmread
from PIL import Image

# Import from your existing codebase
try:
    from block_jacobi import block_jacobi_preconditioner, FRACTION_CLASSES
except ImportError:
    print("WARNING: Could not import block_jacobi. Using inline implementation.")
    
    FRACTION_CLASSES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    
    from scipy.sparse import bmat, csc_matrix
    from scipy.sparse.linalg import inv
    
    def block_jacobi_preconditioner(A, block_size):
        n = A.shape[0]
        inv_blocks = []
        for row_start in range(0, n, block_size):
            row_end = min(row_start + block_size, n)
            block = A[row_start:row_end, row_start:row_end]
            block = csc_matrix(block)
            inv_blocks.append(inv(block))
        
        num_blocks = len(inv_blocks)
        M = bmat([[inv_blocks[i] if i == j else None for j in range(num_blocks)] 
                  for i in range(num_blocks)], format="csr")
        return M


def parse_args():
    parser = argparse.ArgumentParser(description="Profile local Matrix Market files")
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", "-i", type=str,
                             help="Directory containing .mtx files")
    input_group.add_argument("--files", "-f", nargs="+",
                             help="Specific .mtx files to process")
    
    # Output
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output directory for results")
    
    # Processing options
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--gmres-maxiter", type=int, default=1000)
    parser.add_argument("--gmres-tol", type=float, default=1e-6)
    parser.add_argument("--max-n", type=int, default=None,
                        help="Skip matrices larger than this")
    parser.add_argument("--skip-existing", action="store_true")
    
    return parser.parse_args()


def solve_and_profile(A, b, candidates, maxiter=1000, tol=1e-6):
    """Run GMRES for all candidate block sizes."""
    import random
    
    results = {}
    SETUP_COEFF = 5e-7
    SOLVE_COEFF = 1e-5
    
    candidates_shuffled = candidates.copy()
    random.shuffle(candidates_shuffled)
    
    for frac in candidates_shuffled:
        block_size = int(A.shape[0] * frac)
        if block_size < 1:
            continue
        
        t_setup_start = time.perf_counter()
        try:
            M = block_jacobi_preconditioner(A, block_size)
            M_nnz = M.nnz
        except Exception as e:
            results[frac] = {"status": "failed_setup", "error": str(e)}
            continue
        t_setup_end = time.perf_counter()
        
        residual_history = []
        def callback(rk):
            residual_history.append(rk)
        
        t_solve_start = time.perf_counter()
        try:
            _, exitCode = gmres(A, b, rtol=tol, M=M, maxiter=maxiter, callback=callback)
        except Exception as e:
            results[frac] = {"status": "failed_solve", "error": str(e)}
            continue
        t_solve_end = time.perf_counter()
        
        iters = len(residual_history)
        num_blocks = A.shape[0] // block_size
        setup_cost = (block_size ** 3) * SETUP_COEFF
        solve_cost = iters * ((A.nnz + M_nnz) * SOLVE_COEFF)
        
        results[frac] = {
            "status": "converged" if exitCode == 0 else "diverged",
            "block_size": block_size,
            "setup_time": t_setup_end - t_setup_start,
            "solve_time": t_solve_end - t_solve_start,
            "num_blocks": num_blocks,
            "preconditioner_nnz": M_nnz,
            "iterations": iters,
            "total_wall": (t_setup_end - t_setup_start) + (t_solve_end - t_solve_start),
            "total_time": setup_cost + solve_cost,
            "final_residual": float(residual_history[-1]) if residual_history else None
        }
    
    return results


def matrix_to_image(A, image_size):
    """Convert sparse matrix to grayscale sparsity image."""
    n = A.shape[0]
    
    if image_size >= n:
        dense_bool = (A.toarray() != 0)
        pixels = (~dense_bool).astype(np.uint8) * 255
        img = Image.fromarray(pixels, mode="L")
        if image_size > n:
            img = img.resize((image_size, image_size), Image.Resampling.NEAREST)
    else:
        A_coo = A.tocoo()
        tile_size = n / image_size
        pixels = np.ones((image_size, image_size), dtype=np.float32)
        
        for row, col in zip(A_coo.row, A_coo.col):
            px_row = min(int(row / tile_size), image_size - 1)
            px_col = min(int(col / tile_size), image_size - 1)
            pixels[px_row, px_col] = 0.0
        
        pixels = (pixels * 255).astype(np.uint8)
        img = Image.fromarray(pixels, mode="L")
    
    return img


def get_interpolated_label(results):
    """Fit parabola to find continuous optimal fraction."""
    x, y = [], []
    for frac, stats in results.items():
        if stats.get('status') != 'converged':
            continue
        x.append(float(frac))
        y.append(stats['total_time'])
    
    if len(x) < 3:
        return x[np.argmin(y)] if x else None
    
    try:
        z = np.polyfit(x, y, 2)
        a, b, c = z
        if a > 0:
            min_x = -b / (2 * a)
            return max(0.05, min(0.40, min_x))
        return x[np.argmin(y)]
    except:
        return x[np.argmin(y)]


def save_results(A, results, matrix_id, output_dir, image_size, save_raw=False, 
                 source_file=None):
    """Save profiling results."""
    image_dir = Path(output_dir) / "images"
    meta_dir = Path(output_dir) / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    if save_raw:
        matrix_dir = Path(output_dir) / "matrices"
        matrix_dir.mkdir(parents=True, exist_ok=True)
    
    valid_results = {k: v for k, v in results.items() 
                     if isinstance(v, dict) and v.get('status') == 'converged'}
    
    if not valid_results:
        best_frac_time = None
        best_frac_iter = None
        interpolated_opt = None
    else:
        best_frac_time = min(valid_results, key=lambda k: valid_results[k]['total_time'])
        best_frac_iter = min(valid_results, key=lambda k: valid_results[k]['iterations'])
        interpolated_opt = get_interpolated_label(valid_results)
    
    # Save image
    img = matrix_to_image(A, image_size)
    img_name = f"matrix_{matrix_id}.png"
    img.save(image_dir / img_name)
    
    # Save raw matrix
    matrix_filename = None
    if save_raw:
        matrix_filename = f"matrix_{matrix_id}.npz"
        sp.save_npz(matrix_dir / matrix_filename, A)
    
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    meta_data = {
        "matrix_id": matrix_id,
        "source_file": str(source_file) if source_file else None,
        "files": {
            "image": img_name,
            "matrix": matrix_filename,
            "image_size": image_size
        },
        "labels": {
            "class_optimal_time": float(best_frac_time) if best_frac_time else None,
            "class_optimal_iterations": float(best_frac_iter) if best_frac_iter else None,
            "regression_interpolated_optimal": float(interpolated_opt) if interpolated_opt else None
        },
        "matrix_properties": {
            "size": int(A.shape[0]),
            "nnz": int(A.nnz),
            "density": float(A.nnz / (A.shape[0] ** 2))
        },
        "performance_data": convert_to_serializable(results),
        "data_source": "local_gmres_profiled"
    }
    
    with open(meta_dir / f"matrix_{matrix_id}.json", 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    return best_frac_time is not None


def main():
    args = parse_args()
    
    # Collect input files
    if args.input:
        input_dir = Path(args.input)
        mtx_files = list(input_dir.glob("**/*.mtx"))
        print(f"Found {len(mtx_files)} .mtx files in {input_dir}")
    else:
        mtx_files = [Path(f) for f in args.files]
        print(f"Processing {len(mtx_files)} specified files")
    
    if not mtx_files:
        print("No .mtx files found!")
        return
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    candidates = FRACTION_CLASSES
    stats = {"success": 0, "failed": 0, "skipped": 0, "too_large": 0}
    
    for i, mtx_file in enumerate(mtx_files):
        matrix_id = mtx_file.stem  # filename without extension
        print(f"[{i+1}/{len(mtx_files)}] Processing {mtx_file.name}")
        
        # Check if already exists
        if args.skip_existing:
            meta_path = output_dir / "metadata" / f"matrix_{matrix_id}.json"
            if meta_path.exists():
                print(f"  Skipping (exists)")
                stats["skipped"] += 1
                continue
        
        # Load matrix
        try:
            A = mmread(str(mtx_file))
            A = csr_matrix(A)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            stats["failed"] += 1
            continue
        
        # Check size
        if args.max_n and A.shape[0] > args.max_n:
            print(f"  Skipping (n={A.shape[0]} > {args.max_n})")
            stats["too_large"] += 1
            continue
        
        # Check square
        if A.shape[0] != A.shape[1]:
            print(f"  Skipping (not square: {A.shape})")
            stats["failed"] += 1
            continue
        
        print(f"  n={A.shape[0]}, nnz={A.nnz}")
        
        # Profile
        b = np.ones(A.shape[0])
        t_start = time.perf_counter()
        try:
            results = solve_and_profile(A, b, candidates, 
                                        maxiter=args.gmres_maxiter,
                                        tol=args.gmres_tol)
        except Exception as e:
            print(f"  FAILED profiling: {e}")
            stats["failed"] += 1
            continue
        t_elapsed = time.perf_counter() - t_start
        
        # Save
        success = save_results(A, results, matrix_id, output_dir,
                               image_size=args.image_size,
                               save_raw=args.save_raw,
                               source_file=mtx_file)
        
        n_converged = sum(1 for r in results.values() 
                         if isinstance(r, dict) and r.get('status') == 'converged')
        
        if success:
            stats["success"] += 1
            valid = {k: v for k, v in results.items() 
                    if isinstance(v, dict) and v.get('status') == 'converged'}
            best = min(valid, key=lambda k: valid[k]['total_time'])
            print(f"  OK: {n_converged}/8 converged, best={best:.2f}, time={t_elapsed:.1f}s")
        else:
            stats["failed"] += 1
            print(f"  PARTIAL: {n_converged}/8 converged")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Success: {stats['success']}, Failed: {stats['failed']}, "
          f"Skipped: {stats['skipped']}, Too large: {stats['too_large']}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
