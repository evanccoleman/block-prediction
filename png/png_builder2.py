import os
import json
import random
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse import save_npz
from scipy.sparse.linalg import gmres

from PIL import Image
import argparse

# Local imports
from block_jacobi import find_best_block_size, FRACTION_CLASSES


random.seed(42)
np.random.seed(42)

def parse_cli():
    parser = argparse.ArgumentParser(description="PNG Builder - Generate Block-Jacobi Training Data")
    parser.add_argument("--size", type=int, default=500, help="Size of matrices to generate")
    parser.add_argument("--name", "-n", type=str, help="Folder name to save to")
    parser.add_argument("--samples", "-s", type=int, default=3000, help="The number of samples")
    parser.add_argument("--save_raw", action="store_true", help="Flag to save full .npz matrix files")
    parser.add_argument("--skip-gmres", action="store_true", 
                        help="Skip GMRES profiling and use theoretical cost model only. "
                             "Much faster for large matrices or large datasets.")
    parser.add_argument("--iter-model", type=str, default="power", 
                        choices=["constant", "linear", "power"],
                        help="Iteration model for theoretical costs: "
                             "'constant' = fixed iterations, "
                             "'linear' = linear decrease with block size, "
                             "'power' = power law (recommended)")
    parser.add_argument("--base-iters", type=int, default=100,
                        help="Base iteration count at block_fraction=0.20 (default: 100)")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Resolution of output PNG images (default: 128). "
                             "Images will be resized to image_size x image_size.")
    parser.add_argument("--block-density", type=float, default=0.8,
                        help="Density of nonzeros within blocks (default: 0.8). "
                             "Reduce for very large matrices to save memory.")
    return parser.parse_args()


def generate_jittered_matrix(size, target_block_fraction, noise_level, block_density=0.8):
    """
    Generates a matrix where the underlying block structure is 'jittered'.
    
    Args:
        size: Matrix dimension (n x n)
        target_block_fraction: Target block size as fraction of n
        noise_level: Off-diagonal noise density
        block_density: Density of nonzeros within blocks (default 0.8)
    
    Returns:
        A: The sparse matrix
        block_starts: List of positions where blocks start (for per-entry prediction)
    """
    rng = np.random.default_rng()

    # Jitter the block size by +/- 20% of the target
    base_block_size = int(size * target_block_fraction)
    jitter = int(base_block_size * 0.2)
    blocks = []
    block_starts = [0]  # First block always starts at 0
    current_dim = 0
    
    while current_dim < size:
        this_block_size = base_block_size + rng.integers(-jitter, jitter + 1)
        this_block_size = max(2, this_block_size)

        if current_dim + this_block_size > size:
            this_block_size = size - current_dim

        b = sp.random(
            this_block_size, this_block_size, density=block_density, format="coo",
            data_rvs=lambda k: rng.uniform(low=-10.0, high=10.0, size=k)
        )
        blocks.append(b)
        current_dim += this_block_size
        if current_dim < size:
            block_starts.append(current_dim)

    A = sp.block_diag(blocks, format='coo')

    if noise_level > 0:
        noise_mask = sp.random(
            size, size, density=noise_level, format="coo",
            data_rvs=lambda k: rng.uniform(low=-0.5, high=0.5, size=k)
        )
        A = A + noise_mask

    A = A.tocsr()

    # FIX: Weak Diagonal
    # Instead of summing the row (which makes dense blocks huge),
    # we set the diagonal relative to the single-element magnitude (~10.0).
    # This ensures that "cutting" a dense block creates a significant error relative to the diagonal.

    # 1. Base diagonal on element size (15 to 25) vs element size (10)
    diag_values = rng.uniform(15.0, 25.0, size=A.shape[0])

    # 2. Add a tiny fraction of row sum to help prevent pure singularity,
    # but keep it small (0.1) so it doesn't dominate.
    off_diag_sum = np.array(np.abs(A).sum(axis=1)).flatten() - np.abs(A.diagonal())
    diag_values += 0.1 * off_diag_sum

    # 3. Randomize signs to create an indefinite matrix (Hard Mode for GMRES)
    diag_signs = rng.choice([-1, 1], size=A.shape[0])
    A.setdiag(diag_values * diag_signs)

    return A, block_starts


def get_interpolated_label(results, setup_coeff, solve_coeff):
    """
    Fits a parabola to the cost data to find the continuous optimal fraction.
    """
    x = []
    y = []

    for frac_str, stats in results.items():
        if stats['status'] != 'converged': continue

        cost = stats['total_time']

        x.append(float(frac_str))
        y.append(cost)

    if len(x) < 3:
        return x[np.argmin(y)]  # Fallback to discrete if not enough points

    try:
        # Fit y = ax^2 + bx + c
        z = np.polyfit(x, y, 2)
        a, b, c = z

        # Find minimum x = -b / (2a)
        if a > 0:  # Convex (valley)
            min_x = -b / (2 * a)
            return max(0.05, min(0.40, min_x))  # Clamp to valid range
        else:
            return x[np.argmin(y)]  # Concave (hill), pick lowest edge
    except:
        return x[np.argmin(y)]


def solve_and_profile(A, b, candidates):
    """
    Runs GMRES for all candidate block sizes and returns FULL stats.
    """
    results = {}

    # Constants for Theoretical Model
    SETUP_COEFF = 5e-7
    SOLVE_COEFF = 1e-5

    # FIX: Shuffle candidates to prevent "0.05 wins ties" bias
    candidates_shuffled = candidates.copy()
    random.shuffle(candidates_shuffled)

    for frac in candidates_shuffled:
        block_size = int(A.shape[0] * frac)
        if block_size < 1: continue

        from block_jacobi import block_jacobi_preconditioner

        # Measure Setup Time
        t_setup_start = time.perf_counter()
        try:
            M = block_jacobi_preconditioner(A, block_size)
            M_nnz = M.nnz
        except:
            results[frac] = {"status": "failed_setup"}
            continue
        t_setup_end = time.perf_counter()


        residual_history = []
        def cb(rk):
            residual_history.append(rk)

        t_solve_start = time.perf_counter()
        # Fast fail maxiter to speed up generation
        _, exitCode = gmres(A, b, rtol=1e-6, M=M, maxiter=1000, callback=cb)
        t_solve_end = time.perf_counter()

        # 3. Calculate Theoretical "Parallel" Cost
        iters = len(residual_history)
        num_blocks = A.shape[0] // block_size
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
            "total_time": setup_cost_theoretical + solve_cost_theoretical
        }

    return results


def estimate_iterations(block_fraction, model="power", base_iters=100):
    """
    Estimate GMRES iterations based on block size fraction.
    
    The key insight: larger blocks → better preconditioning → fewer iterations.
    
    Models:
    - "constant": All block sizes use base_iters (unrealistic but simple)
    - "linear": Linear decrease from 0.05 to 0.40
    - "power": Power law decay (most realistic based on condition number theory)
    
    The "power" model is based on the observation that:
    - GMRES iterations ∝ sqrt(condition_number)
    - Block-Jacobi condition number ≈ original_cond / block_quality
    - block_quality improves roughly as block_fraction^α for some α
    
    Args:
        block_fraction: Block size as fraction of n (0.05 to 0.40)
        model: "constant", "linear", or "power"
        base_iters: Iteration count at reference point (fraction=0.20)
        
    Returns:
        Estimated iteration count
    """
    # Reference point: block_fraction = 0.20 gives base_iters
    ref_fraction = 0.20
    
    if model == "constant":
        return base_iters
    
    elif model == "linear":
        # Linear: 0.05 → 2x base_iters, 0.40 → 0.5x base_iters
        # iters = base_iters * (1 + slope * (ref_fraction - block_fraction))
        # At 0.05: iters = 2 * base_iters
        # At 0.40: iters = 0.5 * base_iters
        slope = 1.0 / (0.40 - 0.05) * 1.5  # 1.5x range over the interval
        iters = base_iters * (1 + slope * (ref_fraction - block_fraction))
        return max(10, int(iters))  # Floor at 10 iterations
    
    elif model == "power":
        # Power law: iters ∝ (block_fraction)^(-α)
        # Calibrated so that:
        #   - At 0.05: ~2x base_iters (weak preconditioning)
        #   - At 0.20: base_iters (reference)
        #   - At 0.40: ~0.6x base_iters (strong preconditioning)
        #
        # Solving: (0.05/0.20)^(-α) = 2 → α = log(2)/log(4) ≈ 0.5
        alpha = 0.5
        iters = base_iters * (block_fraction / ref_fraction) ** (-alpha)
        return max(10, int(iters))
    
    else:
        raise ValueError(f"Unknown iteration model: {model}")


def compute_theoretical_costs(A, candidates, iter_model="power", base_iters=100):
    """
    Compute theoretical costs WITHOUT running GMRES.
    
    This is much faster and allows generating large datasets or large matrices.
    
    The theoretical model:
    - Setup cost: O(block_size^3) for inverting the largest block
    - Solve cost: estimated_iterations * O(nnz) for SpMV operations
    
    The iteration count is estimated based on block size using the specified model,
    capturing the key insight that larger blocks provide better preconditioning.
    
    Args:
        A: Sparse matrix
        candidates: List of block fraction candidates
        iter_model: "constant", "linear", or "power"
        base_iters: Base iteration count at block_fraction=0.20
        
    Returns:
        results: Dict mapping fraction -> cost info
    """
    results = {}
    
    # Constants for Theoretical Model (same as solve_and_profile)
    SETUP_COEFF = 5e-7
    SOLVE_COEFF = 1e-5
    
    n = A.shape[0]
    A_nnz = A.nnz
    
    for frac in candidates:
        block_size = int(n * frac)
        if block_size < 1: 
            continue
        
        # Estimate iterations based on block size
        estimated_iters = estimate_iterations(frac, model=iter_model, base_iters=base_iters)
        
        # Theoretical preconditioner nnz (block diagonal with dense blocks)
        num_blocks = n // block_size
        remainder = n % block_size
        
        # Each block is dense: block_size^2 entries
        # Plus remainder block if any
        M_nnz_estimate = num_blocks * (block_size ** 2)
        if remainder > 0:
            M_nnz_estimate += remainder ** 2
        
        # Theoretical costs
        setup_cost = (block_size ** 3) * SETUP_COEFF
        solve_cost = estimated_iters * ((A_nnz + M_nnz_estimate) * SOLVE_COEFF)
        
        results[frac] = {
            "status": "theoretical",
            "block_size": block_size,
            "num_blocks": num_blocks,
            "preconditioner_nnz_estimate": M_nnz_estimate,
            "iterations_estimated": estimated_iters,
            "iteration_model": iter_model,
            "setup_cost_theoretical": setup_cost,
            "solve_cost_theoretical": solve_cost,
            "total_time": setup_cost + solve_cost
        }
    
    return results


def get_theoretical_optimal(results):
    """
    Find the optimal block fraction from theoretical costs.
    """
    if not results:
        return 0.20  # Default fallback
    
    # Find fraction with minimum total_time
    best_frac = min(results.keys(), key=lambda k: results[k]['total_time'])
    return best_frac


def get_interpolated_label_theoretical(results):
    """
    Fits a parabola to theoretical cost data to find continuous optimal.
    """
    x = []
    y = []
    
    for frac, stats in results.items():
        x.append(float(frac))
        y.append(stats['total_time'])
    
    if len(x) < 3:
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


def generate_sparse(block_sizes, noise=0.1, de_noise=0.1, random_state=42):
    rng = np.random.default_rng(random_state)
    blocks = []

    # Generate dense blocks
    for block in block_sizes:
        blocks.append(
            sp.random(
                block,
                block,
                density=1.0,
                format="coo",
                data_rvs=lambda k: rng.uniform(low=-0.10, high=0.0, size=k),
            )
        )

    # Create block diagonal structure
    A = sp.block_diag(blocks, format='coo')
    n = A.shape[0]

    # Add noise
    if noise > 0.0:
        noise_mask = sp.random(
            n,
            n,
            density=noise,
            format="coo",
            data_rvs=lambda k: rng.uniform(low=-0.10, high=0.0, size=k),
        )
        A = A + noise_mask

    # Apply de-noise (drop random elements)
    if de_noise > 0.0:
        A = A.tocoo()
        keep_mask = rng.random(A.nnz) >= de_noise
        A = sp.coo_matrix(
            (A.data[keep_mask], (A.row[keep_mask], A.col[keep_mask])),
            shape=A.shape
        )

    # Convert to CSR before setting diagonal to ensure efficiency and correctness
    A = A.tocsr()
    A.setdiag(100.0)
    return A


def generate_acceptable_blocks(matrix_size):
    block_list = []

    # all divisors of matrix size
    divisors = FRACTION_CLASSES

    # taking all divisors greater than 2% of matrix size but less than 30%
    for divisor in divisors:
            block_list.append(int(divisor*matrix_size))

    return block_list


def matrix_to_png(matrix):
    # 255 is white, 0 is black.
    # when we use matplotlib to show matrices, we view the 1s as black so we need to invert matrix
    dense_bool = matrix.toarray() != 0
    pixels     = (~dense_bool).astype(np.uint8) * 255
    img = Image.fromarray(pixels, mode="L")
    return img


def store_pngs(data, labels, width, height, base_dir='png_dataset128'):
    os.makedirs(base_dir, exist_ok=True)

    # Save the list of classes for reference
    matrix_size = data[0].shape[0]
    classes = [int(i * matrix_size) for i in FRACTION_CLASSES]


    class_path = os.path.join(base_dir, 'classes.txt')
    with open(class_path, "w") as f:
        for elem in classes:
            f.write(f"{elem}\n")

    for label in labels:
        label_folder = os.path.join(base_dir, f'label_{label}')
        os.makedirs(label_folder, exist_ok=True)

    for i in range(len(data)):
        matrix = data[i]
        img = matrix_to_png(matrix)
        file_path = os.path.join(base_dir, f'label_{labels[i]}',f'matrix_{i}.png')
        img.save(file_path)


def convert_to_serializable(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def sparse_to_image(matrix, image_size):
    """
    Convert a sparse matrix to a fixed-size grayscale image WITHOUT
    converting to dense format. This is O(nnz) memory instead of O(n²).
    
    Each pixel represents the density of nonzeros in a tile of the matrix.
    White (255) = empty tile, Black (0) = fully dense tile.
    
    Args:
        matrix: scipy sparse matrix
        image_size: output image dimension (image_size × image_size)
        
    Returns:
        PIL Image
    """
    n = matrix.shape[0]
    tile_size = n / image_size  # Can be fractional
    
    # Convert to COO format for easy access to row/col indices
    coo = matrix.tocoo()
    
    # Map each nonzero to its pixel location
    pixel_rows = (coo.row / tile_size).astype(np.int32)
    pixel_cols = (coo.col / tile_size).astype(np.int32)
    
    # Clamp to valid range (handles edge case where index == n)
    pixel_rows = np.clip(pixel_rows, 0, image_size - 1)
    pixel_cols = np.clip(pixel_cols, 0, image_size - 1)
    
    # Count nonzeros per pixel using 2D histogram
    density, _, _ = np.histogram2d(
        pixel_rows, pixel_cols,
        bins=[np.arange(image_size + 1), np.arange(image_size + 1)]
    )
    
    # Normalize: max possible nonzeros per tile = tile_size²
    max_per_tile = tile_size * tile_size
    density_normalized = density / max_per_tile
    
    # Convert to grayscale: 0 density = white (255), 1 density = black (0)
    pixels = ((1 - density_normalized) * 255).astype(np.uint8)
    
    return Image.fromarray(pixels, mode="L")


def save_data(matrix, results, matrix_id, base_dir, target_structure, block_starts=None, 
              save_raw=False, is_theoretical=False, image_size=128):
    """
    Saves PNG, Raw Matrix (.npz), and JSON Sidecar.
    
    Args:
        matrix: Sparse matrix
        results: Performance results (from GMRES or theoretical)
        matrix_id: Unique ID for this matrix
        base_dir: Output directory
        target_structure: The target block fraction used in generation
        block_starts: List of block start positions (for per-entry labels)
        save_raw: Whether to save the raw .npz matrix
        is_theoretical: Whether results are from theoretical model (no GMRES)
        image_size: Resolution of output PNG (image_size x image_size)
    """
    # 1. Determine discrete "Optimal" Label (classification)
    if is_theoretical:
        # All results are valid for theoretical
        valid_results = results
    else:
        valid_results = {k: v for k, v in results.items() if v['status'] == 'converged'}

    if not valid_results:
        print(f"Matrix {matrix_id} did not converge. Skipping.")
        return

    best_frac_time = min(valid_results, key=lambda k: valid_results[k]['total_time'])
    
    if is_theoretical:
        best_frac_iter = best_frac_time  # No iteration data, use same as time
        interpolated_opt = get_interpolated_label_theoretical(valid_results)
    else:
        best_frac_iter = min(valid_results, key=lambda k: valid_results[k]['iterations'])
        SETUP_COEFF = 5e-7
        SOLVE_COEFF = 1e-5
        interpolated_opt = get_interpolated_label(valid_results, SETUP_COEFF, SOLVE_COEFF)

    # 2. Setup directories
    image_dir = os.path.join(base_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    meta_dir = os.path.join(base_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    if save_raw:
        matrix_dir = os.path.join(base_dir, "matrices")
        os.makedirs(matrix_dir, exist_ok=True)

    # 3. Save Image (Visual Representation)
    # IMPORTANT: We must NOT call matrix.toarray() for large matrices!
    # Instead, compute pixel densities directly from sparse structure.
    native_size = matrix.shape[0]
    
    if image_size >= native_size:
        # Small matrix: safe to convert to dense
        dense_bool = matrix.toarray() != 0
        pixels = (~dense_bool).astype(np.uint8) * 255
        img = Image.fromarray(pixels, mode="L")
        if image_size > native_size:
            img = img.resize((image_size, image_size), Image.Resampling.NEAREST)
    else:
        # Large matrix: compute tile densities without materializing dense array
        # Each pixel in output represents a (tile_size × tile_size) region
        img = sparse_to_image(matrix, image_size)
    
    img_name = f"matrix_{matrix_id}.png"
    img.save(os.path.join(image_dir, img_name))

    # 4. Save Raw Matrix (Data Representation, conditional)
    matrix_filename = None
    if save_raw:
        matrix_filename = f"matrix_{matrix_id}.npz"
        save_npz(os.path.join(matrix_dir, matrix_filename), matrix)

    # 5. Save Metadata
    meta_data = {
        "matrix_id": int(matrix_id),
        "files": {
            "image": img_name,
            "matrix": matrix_filename,
            "image_size": int(image_size)
        },
        "ground_truth": {
            "generated_structure_fraction": float(target_structure)
        },
        "labels": {
            # CLASSIFICATION LABELS (Buckets)
            "class_optimal_time": float(best_frac_time),
            "class_optimal_iterations": float(best_frac_iter),

            # REGRESSION LABELS (Floats)
            "regression_ground_truth": float(target_structure),       # The Physics Truth
            "regression_interpolated_optimal": float(interpolated_opt) # The Solver Truth
        },
        "matrix_properties": {
            "size": int(matrix.shape[0]),
            "nnz": int(matrix.nnz),
            "density": float(matrix.nnz / (matrix.shape[0]**2))
        },
        "performance_data": convert_to_serializable(results),
        "data_source": "theoretical" if is_theoretical else "gmres_profiled"
    }
    
    # Add block starts for per-entry prediction training
    if block_starts is not None:
        meta_data["block_starts"] = [int(x) for x in block_starts]

    with open(os.path.join(meta_dir, f"matrix_{matrix_id}.json"), 'w') as f:
        json.dump(meta_data, f, indent=2)


def generate_matrices(size, folder_name, sample_amount=3000):
    """Legacy function - generates matrices with GMRES profiling."""
    best_size_array = []
    dataset = []

    acceptable_blocks = generate_acceptable_blocks(size)
    print(f"acceptable blocks for size {size}: {acceptable_blocks}")
    
    block_samples = sample_amount // len(acceptable_blocks)
    for block in acceptable_blocks:
        for _ in range(block_samples):
            blocks = []
            num_blocks = size // block
            remainder = size % block

            for k in range(num_blocks):
                blocks.append(block)
            if remainder != 0:
                blocks.append(remainder)

            A = generate_sparse(blocks, noise=0.01, de_noise=0.01)
            dataset.append(A)

            n = A.shape[0]
            b = np.ones(A.shape[0])
        
            best_size = find_best_block_size(n, A, b)
            best_size_array.append(best_size)

    store_pngs(dataset, best_size_array, 500, 500, base_dir=folder_name)

    return


def generate_pipeline(size, folder_name, samples, save_raw=False, 
                      skip_gmres=False, iter_model="power", base_iters=100,
                      image_size=128, block_density=0.8):
    """
    Main generation pipeline.
    
    Args:
        size: Matrix size (n x n)
        folder_name: Output directory
        samples: Number of samples to generate
        save_raw: Whether to save raw .npz matrices
        skip_gmres: If True, skip GMRES profiling and use theoretical costs only
        iter_model: Iteration model for theoretical costs ("constant", "linear", "power")
        base_iters: Base iteration count at block_fraction=0.20
        image_size: Resolution of output PNG images (default 128x128)
        block_density: Density of nonzeros within blocks (default 0.8)
    """
    candidates = FRACTION_CLASSES
    
    mode_str = "THEORETICAL ONLY" if skip_gmres else "GMRES PROFILED"
    print(f"Generation mode: {mode_str}")
    print(f"Matrix size: {size}x{size}")
    print(f"Image output size: {image_size}x{image_size}")
    print(f"Block density: {block_density}")
    print(f"Samples: {samples}")
    
    # Estimate memory requirements
    avg_block_size = size * 0.225  # Average block fraction
    num_blocks = size / avg_block_size
    block_nnz = num_blocks * (block_density * avg_block_size ** 2)
    noise_nnz = 0.01 * size * size  # Average noise density
    total_nnz = block_nnz + noise_nnz
    # COO format: row (4B) + col (4B) + data (8B) = 16 bytes per nnz
    mem_gb = (total_nnz * 16) / (1024**3)
    print(f"Estimated memory per matrix: ~{mem_gb:.1f} GB (nnz ≈ {total_nnz/1e6:.0f}M)")
    
    if mem_gb > 16:
        print(f"WARNING: Large memory requirement! Consider:")
        print(f"  - Reducing matrix size")
        print(f"  - Reducing --block-density (currently {block_density})")
        print(f"  - Using a high-memory node")
    
    if skip_gmres:
        print(f"Iteration model: {iter_model}")
        print(f"Base iterations (at 20%): {base_iters}")
        # Show iteration estimates for each candidate
        print(f"Estimated iterations by block fraction:")
        for frac in sorted(candidates):
            est_iters = estimate_iterations(frac, model=iter_model, base_iters=base_iters)
            print(f"  {frac:.2f} -> {est_iters} iterations")
    print()

    for i in range(samples):
        if i % 10 == 0: 
            print(f"Generating sample {i}/{samples}...")

        # Jittered generation
        target_structure = random.uniform(0.05, 0.40)
        noise = random.uniform(0.001, 0.02)

        A, block_starts = generate_jittered_matrix(size, target_structure, noise, 
                                                    block_density=block_density)
        b = np.ones(A.shape[0])

        if skip_gmres:
            # Use theoretical cost model only
            results = compute_theoretical_costs(A, candidates, 
                                                iter_model=iter_model, 
                                                base_iters=base_iters)
            save_data(A, results, i, folder_name, target_structure, 
                     block_starts=block_starts, save_raw=save_raw, is_theoretical=True,
                     image_size=image_size)
        else:
            # Run actual GMRES profiling
            results = solve_and_profile(A, b, candidates)
            save_data(A, results, i, folder_name, target_structure,
                     block_starts=block_starts, save_raw=save_raw, is_theoretical=False,
                     image_size=image_size)


if __name__ == '__main__':
    args = parse_cli()
    
    generate_pipeline(
        args.size, 
        args.name, 
        args.samples, 
        save_raw=args.save_raw,
        skip_gmres=args.skip_gmres,
        iter_model=args.iter_model,
        base_iters=args.base_iters,
        image_size=args.image_size,
        block_density=args.block_density
    )
