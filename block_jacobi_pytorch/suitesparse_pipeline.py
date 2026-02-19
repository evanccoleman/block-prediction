#!/usr/bin/env python3
"""
SuiteSparse Evaluation Pipeline

Downloads SPD matrices from SuiteSparse Matrix Collection, profiles them
with GMRES + block-Jacobi preconditioning, and saves results in the same
format as png_builder2.py for model evaluation.

Usage:
    python suitesparse_pipeline.py --output ./suitesparse_eval --max-matrices 50
    python suitesparse_pipeline.py --output ./suitesparse_eval --max-n 500 --min-n 100
    python suitesparse_pipeline.py --list-only  # Just show what would be downloaded

Requirements:
    pip install ssgetpy scipy numpy pillow
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres
from scipy.io import mmread
from PIL import Image

# Try to import ssgetpy
try:
    import ssgetpy
except ImportError:
    print("ERROR: ssgetpy not installed. Run: pip install ssgetpy")
    sys.exit(1)

# Import from your existing codebase
# Adjust this path if needed
try:
    from block_jacobi import block_jacobi_preconditioner, FRACTION_CLASSES
except ImportError:
    print("WARNING: Could not import block_jacobi. Using inline implementation.")
    
    FRACTION_CLASSES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    
    from scipy.sparse import bmat, csc_matrix
    from scipy.sparse.linalg import inv
    
    def block_jacobi_preconditioner(A, block_size):
        """Creates a block Jacobi preconditioner for a sparse matrix A."""
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
    parser = argparse.ArgumentParser(
        description="Download and profile SuiteSparse matrices for block-Jacobi evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download up to 50 SPD matrices with n <= 1000
    python suitesparse_pipeline.py --output ./suitesparse_eval --max-matrices 50

    # Only matrices between 100 and 500
    python suitesparse_pipeline.py --output ./suitesparse_eval --min-n 100 --max-n 500

    # Just list what would be downloaded (no download)
    python suitesparse_pipeline.py --list-only --max-n 1000

    # Include non-SPD matrices (general square)
    python suitesparse_pipeline.py --output ./suitesparse_eval --include-non-spd
        """
    )
    
    # Output settings
    parser.add_argument("--output", "-o", type=str, default="./suitesparse_eval",
                        help="Output directory for results (default: ./suitesparse_eval)")
    parser.add_argument("--cache-dir", type=str, default="./suitesparse_cache",
                        help="Directory to cache downloaded matrices (default: ./suitesparse_cache)")
    
    # Matrix selection
    parser.add_argument("--max-n", type=int, default=1000,
                        help="Maximum matrix dimension (default: 1000)")
    parser.add_argument("--min-n", type=int, default=10,
                        help="Minimum matrix dimension (default: 10)")
    parser.add_argument("--max-matrices", type=int, default=None,
                        help="Maximum number of matrices to process (default: all matching)")
    parser.add_argument("--include-non-spd", action="store_true",
                        help="Include non-SPD matrices (default: SPD only)")
    parser.add_argument("--groups", type=str, nargs="*", default=None,
                        help="Specific matrix groups to include (e.g., HB Boeing BCSSTK)")
    
    # Processing settings
    parser.add_argument("--image-size", type=int, default=128,
                        help="Output image resolution (default: 128)")
    parser.add_argument("--save-raw", action="store_true",
                        help="Also save raw .npz matrix files")
    parser.add_argument("--gmres-maxiter", type=int, default=1000,
                        help="Maximum GMRES iterations (default: 1000)")
    parser.add_argument("--gmres-tol", type=float, default=1e-6,
                        help="GMRES convergence tolerance (default: 1e-6)")
    
    # Utility options
    parser.add_argument("--list-only", action="store_true",
                        help="Only list matching matrices, don't download or process")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip matrices that already have results")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    return parser.parse_args()


def query_suitesparse(min_n, max_n, spd_only=True, groups=None):
    """
    Query SuiteSparse for matrices matching criteria.
    
    Returns list of matrix metadata dictionaries.
    """
    print(f"Querying SuiteSparse for matrices with {min_n} <= n <= {max_n}...")
    
    # ssgetpy search returns a list of Matrix objects
    # We filter by size and properties
    
    # Get all matrices in size range
    results = ssgetpy.search(
        nrows=(min_n, max_n),
        ncols=(min_n, max_n),
        isspd=True if spd_only else None,
        # Only square matrices
    )
    
    # Filter for square matrices
    results = [m for m in results if m.nrows == m.ncols]
    
    # Filter by group if specified
    if groups:
        groups_lower = [g.lower() for g in groups]
        results = [m for m in results if m.group.lower() in groups_lower]
    
    # Filter for real-valued matrices (not complex)
    results = [m for m in results if not m.isComplex]
    
    print(f"Found {len(results)} matching matrices")
    
    return results


def download_matrix(matrix_info, cache_dir):
    """
    Download a matrix from SuiteSparse, using cache if available.
    
    Returns scipy sparse matrix or None if download fails.
    """
    cache_path = Path(cache_dir) / f"{matrix_info.group}_{matrix_info.name}.mtx"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        # Load from cache
        try:
            A = mmread(str(cache_path))
            return csr_matrix(A)
        except Exception as e:
            print(f"  Cache read failed: {e}, re-downloading...")
    
    # Download using ssgetpy
    try:
        # ssgetpy.fetch downloads to a temp location
        matrix_info.download(destpath=str(cache_path.parent), extract=True)
        
        # Find the .mtx file (ssgetpy extracts to a subdirectory)
        extract_dir = cache_path.parent / matrix_info.name
        mtx_file = extract_dir / f"{matrix_info.name}.mtx"
        
        if mtx_file.exists():
            A = mmread(str(mtx_file))
            # Copy to our cache location for future use
            import shutil
            shutil.copy(mtx_file, cache_path)
            return csr_matrix(A)
        else:
            # Try alternative locations
            for pattern in [f"{matrix_info.name}.mtx", "*.mtx"]:
                import glob
                matches = glob.glob(str(extract_dir / "**" / pattern), recursive=True)
                if matches:
                    A = mmread(matches[0])
                    shutil.copy(matches[0], cache_path)
                    return csr_matrix(A)
            
            print(f"  Could not find .mtx file in {extract_dir}")
            return None
            
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def solve_and_profile(A, b, candidates, maxiter=1000, tol=1e-6):
    """
    Runs GMRES for all candidate block sizes and returns stats.
    
    This mirrors the function in png_builder2.py.
    """
    import random
    
    results = {}
    
    # Constants for theoretical cost model (same as png_builder2.py)
    SETUP_COEFF = 5e-7
    SOLVE_COEFF = 1e-5
    
    # Shuffle to prevent bias
    candidates_shuffled = candidates.copy()
    random.shuffle(candidates_shuffled)
    
    for frac in candidates_shuffled:
        block_size = int(A.shape[0] * frac)
        if block_size < 1:
            continue
        
        # Measure setup time
        t_setup_start = time.perf_counter()
        try:
            M = block_jacobi_preconditioner(A, block_size)
            M_nnz = M.nnz
        except Exception as e:
            results[frac] = {"status": "failed_setup", "error": str(e)}
            continue
        t_setup_end = time.perf_counter()
        
        # Run GMRES
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
        
        # Theoretical parallel cost
        setup_cost_theoretical = (block_size ** 3) * SETUP_COEFF
        solve_cost_theoretical = iters * ((A.nnz + M_nnz) * SOLVE_COEFF)
        
        results[frac] = {
            "status": "converged" if exitCode == 0 else "diverged",
            "block_size": block_size,
            "setup_time": t_setup_end - t_setup_start,
            "solve_time": t_solve_end - t_solve_start,
            "num_blocks": num_blocks,
            "preconditioner_nnz": M_nnz,
            "iterations": iters,
            "total_wall": (t_setup_end - t_setup_start) + (t_solve_end - t_solve_start),
            "total_time": setup_cost_theoretical + solve_cost_theoretical,
            "final_residual": float(residual_history[-1]) if residual_history else None
        }
    
    return results


def matrix_to_image(A, image_size):
    """
    Convert sparse matrix to grayscale sparsity image.
    
    White = zero, Black = nonzero (matches matplotlib spy convention inverted for PNG)
    """
    n = A.shape[0]
    
    if image_size >= n:
        # Small matrix: direct conversion
        dense_bool = (A.toarray() != 0)
        pixels = (~dense_bool).astype(np.uint8) * 255
        img = Image.fromarray(pixels, mode="L")
        if image_size > n:
            img = img.resize((image_size, image_size), Image.Resampling.NEAREST)
    else:
        # Large matrix: compute tile densities
        A_coo = A.tocoo()
        tile_size = n / image_size
        
        pixels = np.ones((image_size, image_size), dtype=np.float32)
        
        for row, col in zip(A_coo.row, A_coo.col):
            px_row = min(int(row / tile_size), image_size - 1)
            px_col = min(int(col / tile_size), image_size - 1)
            pixels[px_row, px_col] = 0.0  # Mark as nonzero
        
        # Could do density-based shading, but binary is simpler
        pixels = (pixels * 255).astype(np.uint8)
        img = Image.fromarray(pixels, mode="L")
    
    return img


def get_interpolated_label(results):
    """
    Fits a parabola to find continuous optimal fraction.
    """
    x, y = [], []
    
    for frac, stats in results.items():
        if stats.get('status') != 'converged':
            continue
        x.append(float(frac))
        y.append(stats['total_time'])
    
    if len(x) < 3:
        if len(x) == 0:
            return None
        return x[np.argmin(y)]
    
    try:
        z = np.polyfit(x, y, 2)
        a, b, c = z
        if a > 0:  # Convex
            min_x = -b / (2 * a)
            return max(0.05, min(0.40, min_x))
        else:
            return x[np.argmin(y)]
    except:
        return x[np.argmin(y)]


def save_results(A, results, matrix_info, output_dir, image_size, save_raw=False):
    """
    Save profiling results in the same format as png_builder2.py.
    """
    # Create directories
    image_dir = Path(output_dir) / "images"
    meta_dir = Path(output_dir) / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    if save_raw:
        matrix_dir = Path(output_dir) / "matrices"
        matrix_dir.mkdir(parents=True, exist_ok=True)
    
    # Matrix identifier
    matrix_id = f"{matrix_info.group}_{matrix_info.name}"
    
    # Filter valid results
    valid_results = {k: v for k, v in results.items() 
                     if isinstance(v, dict) and v.get('status') == 'converged'}
    
    if not valid_results:
        print(f"  WARNING: No block sizes converged for {matrix_id}")
        # Still save metadata to record the failure
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
    
    # Save raw matrix if requested
    matrix_filename = None
    if save_raw:
        matrix_filename = f"matrix_{matrix_id}.npz"
        sp.save_npz(matrix_dir / matrix_filename, A)
    
    # Convert results for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Build metadata
    meta_data = {
        "matrix_id": matrix_id,
        "suitesparse_info": {
            "group": matrix_info.group,
            "name": matrix_info.name,
            "id": matrix_info.id,
            "kind": getattr(matrix_info, 'kind', None),
            "is_spd": matrix_info.isspd,
        },
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
        "data_source": "suitesparse_gmres_profiled"
    }
    
    with open(meta_dir / f"matrix_{matrix_id}.json", 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    return best_frac_time is not None


def main():
    args = parse_args()
    
    # Query SuiteSparse
    matrices = query_suitesparse(
        min_n=args.min_n,
        max_n=args.max_n,
        spd_only=not args.include_non_spd,
        groups=args.groups
    )
    
    if not matrices:
        print("No matrices found matching criteria.")
        return
    
    # Limit number of matrices
    if args.max_matrices and len(matrices) > args.max_matrices:
        print(f"Limiting to first {args.max_matrices} matrices")
        matrices = matrices[:args.max_matrices]
    
    # List-only mode
    if args.list_only:
        print(f"\n{'='*60}")
        print(f"{'ID':>6} | {'Group':<15} | {'Name':<20} | {'N':>6} | {'NNZ':>10}")
        print(f"{'='*60}")
        for m in matrices:
            print(f"{m.id:>6} | {m.group:<15} | {m.name:<20} | {m.nrows:>6} | {m.nnz:>10}")
        print(f"{'='*60}")
        print(f"Total: {len(matrices)} matrices")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process matrices
    print(f"\nProcessing {len(matrices)} matrices...")
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {args.cache_dir}")
    print()
    
    candidates = FRACTION_CLASSES
    stats = {"success": 0, "failed_download": 0, "failed_profile": 0, "skipped": 0}
    
    for i, matrix_info in enumerate(matrices):
        matrix_id = f"{matrix_info.group}_{matrix_info.name}"
        print(f"[{i+1}/{len(matrices)}] Processing {matrix_id} (n={matrix_info.nrows}, nnz={matrix_info.nnz})")
        
        # Check if already processed
        if args.skip_existing:
            meta_path = output_dir / "metadata" / f"matrix_{matrix_id}.json"
            if meta_path.exists():
                print(f"  Skipping (already exists)")
                stats["skipped"] += 1
                continue
        
        # Download matrix
        A = download_matrix(matrix_info, args.cache_dir)
        if A is None:
            print(f"  FAILED: Could not download")
            stats["failed_download"] += 1
            continue
        
        # Ensure CSR format
        A = csr_matrix(A)
        
        # Create RHS vector
        b = np.ones(A.shape[0])
        
        # Profile with GMRES
        t_start = time.perf_counter()
        try:
            results = solve_and_profile(
                A, b, candidates,
                maxiter=args.gmres_maxiter,
                tol=args.gmres_tol
            )
        except Exception as e:
            print(f"  FAILED: Profiling error: {e}")
            stats["failed_profile"] += 1
            continue
        t_elapsed = time.perf_counter() - t_start
        
        # Count converged
        n_converged = sum(1 for r in results.values() 
                         if isinstance(r, dict) and r.get('status') == 'converged')
        
        # Save results
        success = save_results(
            A, results, matrix_info, output_dir,
            image_size=args.image_size,
            save_raw=args.save_raw
        )
        
        if success:
            stats["success"] += 1
            # Find best block size for reporting
            valid = {k: v for k, v in results.items() 
                    if isinstance(v, dict) and v.get('status') == 'converged'}
            if valid:
                best = min(valid, key=lambda k: valid[k]['total_time'])
                print(f"  OK: {n_converged}/{len(candidates)} converged, "
                      f"best={best:.2f}, time={t_elapsed:.1f}s")
        else:
            stats["failed_profile"] += 1
            print(f"  PARTIAL: {n_converged}/{len(candidates)} converged, time={t_elapsed:.1f}s")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Successful:       {stats['success']}")
    print(f"  Failed download:  {stats['failed_download']}")
    print(f"  Failed profiling: {stats['failed_profile']}")
    print(f"  Skipped:          {stats['skipped']}")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    
    # Save summary JSON
    summary = {
        "query_params": {
            "min_n": args.min_n,
            "max_n": args.max_n,
            "spd_only": not args.include_non_spd,
            "groups": args.groups
        },
        "processing_params": {
            "image_size": args.image_size,
            "gmres_maxiter": args.gmres_maxiter,
            "gmres_tol": args.gmres_tol,
            "candidates": candidates
        },
        "stats": stats,
        "matrices_processed": [f"{m.group}_{m.name}" for m in matrices]
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
