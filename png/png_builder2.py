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
    parser = argparse.ArgumentParser(description="PNG Builder")
    parser.add_argument("--size", type=int, default=500, help="Size of matrices to generate")
    parser.add_argument("--name", "-n", type=str, help="Folder name to save to")
    parser.add_argument("--samples", "-s", type=int, default=3000, help="The number of samples")
    return parser.parse_args()


def generate_jittered_matrix(size, target_block_fraction, noise_level):
    """
    Generates a matrix where the underlying block structure is 'jittered'.
    """
    rng = np.random.default_rng()

    # Jitter the block size by +/- 20% of the target
    base_block_size = int(size * target_block_fraction)
    jitter = int(base_block_size * 0.2)

    blocks = []
    current_dim = 0
    while current_dim < size:
        this_block_size = base_block_size + rng.integers(-jitter, jitter + 1)
        this_block_size = max(2, this_block_size)

        if current_dim + this_block_size > size:
            this_block_size = size - current_dim

        b = sp.random(
            this_block_size, this_block_size, density=0.8, format="coo",
            data_rvs=lambda k: rng.uniform(low=-10.0, high=10.0, size=k)
        )
        blocks.append(b)
        current_dim += this_block_size

    A = sp.block_diag(blocks, format='coo')

    if noise_level > 0:
        noise_mask = sp.random(
            size, size, density=noise_level, format="coo",
            data_rvs=lambda k: rng.uniform(low=-0.5, high=0.5, size=k)
        )
        A = A + noise_mask

    A = A.tocsr()
    #A.setdiag(100.0)

    # # NEW: Dynamic Diagonal Dominance
    # # Calculate the sum of absolute off-diagonal values for each row
    # off_diag_sum = np.array(np.abs(A).sum(axis=1)).flatten() - np.abs(A.diagonal())
    #
    # # Set diagonal to be just barely dominant or slightly weak (1.01x to 1.5x the noise)
    # # This forces the solver to rely on the block structure
    # diag_values = off_diag_sum * rng.uniform(1.01, 2.5, size=A.shape[0])
    #
    # # Add a safety floor to prevent absolute singularities
    # diag_values += 0.5

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

    A.setdiag(diag_values)
    return A


def solve_and_profile(A, b, candidates):
    """
    Runs GMRES for all candidate block sizes and returns FULL stats.
    """
    results = {}

    # Constants for Theoretical Model
    # We tune these so setup_cost and solve_cost are roughly balanced
    # for a "medium" block size (e.g., 0.20)
    SETUP_COEFF = 1e-7
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
        _, exitCode = gmres(A, b, rtol=1e-2, M=M, maxiter=1000, callback=cb)
        t_solve_end = time.perf_counter()

        # 3. Calculate Theoretical "Parallel" Cost
        iters = len(residual_history)
        num_blocks = A.shape[0] // block_size
        # We assume enough GPU cores to parallelize the blocks,
        # so the bottleneck is the inversion of a SINGLE block of size m
        # Complexity: O(m^3)
        setup_cost_theoretical = (block_size ** 3) * SETUP_COEFF

        # Solve cost: Iterations * Cost of SpMV (proportional to NNZ)
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

'''
Function converts matrix to png format
Parameters:
    - matrix: The matrix to convert.
Returns the PIL image object.
'''
def matrix_to_png(matrix):
    # 255 is white, 0 is black.
    # when we use matplotlib to show matrices, we view the 1s as black so we need to invert matrix
    dense_bool = matrix.toarray() != 0
    pixels     = (~dense_bool).astype(np.uint8) * 255
    img = Image.fromarray(pixels, mode="L")
    #img.show()
    return img

'''
Function stores the pngs to a file directory.
Parameters:
    - data: The array of matrices.
    - labels: The block sizes.
    - width: The width of the pngs. By default, pass the matrix size.
    - height: The height of the pngs. By default, pass the matrix size.
    - base_dir: Where to store the pngs.
'''
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

        #fig, ax = plt.subplots()
        #ax.set_axis_off()
        #plt.savefig(file_path, bbox_inches='tight', pad_inches=0)img = matrix_to_png(matrix)
        #file_path = os.path.join(size_folder, f'label_{labels[i]}',f'matrix_{i}.png')
        #img.save(file_path)

        #fig, ax = plt.subplots()
        #ax.set_axis_off()
        #plt.savefig(file_path, bbox_inches='tight', pad_inches=0)


def save_data(matrix, results, matrix_id, base_dir, target_structure):
    """
    Saves PNG, Raw Matrix (.npz), and JSON Sidecar.
    Structure:
      base_dir/
        images/     -> Visuals for CNN
        matrices/   -> Raw .npz for dense/diagonal experiments
        metadata/   -> Labels and stats
    """
    # 1. Determine "Optimal" Label
    valid_results = {k: v for k, v in results.items() if v['status'] == 'converged'}

    if not valid_results:
        print(f"Matrix {matrix_id} did not converge. Skipping.")
        return

    best_frac_time = min(valid_results, key=lambda k: valid_results[k]['total_time'])
    best_frac_iter = min(valid_results, key=lambda k: valid_results[k]['iterations'])

    # 2. Setup directories
    image_dir = os.path.join(base_dir, "images")
    matrix_dir = os.path.join(base_dir, "matrices") # New Folder
    meta_dir = os.path.join(base_dir, "metadata")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(matrix_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # 3. Save Image (Visual Representation)
    dense_bool = matrix.toarray() != 0
    pixels = (~dense_bool).astype(np.uint8) * 255
    img = Image.fromarray(pixels, mode="L")
    img_name = f"matrix_{matrix_id}.png"
    img.save(os.path.join(image_dir, img_name))

    # 4. Save Raw Matrix (Data Representation)
    matrix_filename = f"matrix_{matrix_id}.npz"
    save_npz(os.path.join(matrix_dir, matrix_filename), matrix)

    # 5. Save Metadata
    meta_data = {
        "matrix_id": matrix_id,
        "files": {
            "image": img_name,
            "matrix": matrix_filename
        },
        "ground_truth": {
            "generated_structure_fraction": target_structure
        },
        "labels": {
            "optimal_fraction_total_time": best_frac_time,
            "optimal_fraction_iterations": best_frac_iter
        },
        "matrix_properties": {
            "size": matrix.shape[0],
            "nnz": matrix.nnz,
            "density": matrix.nnz / (matrix.shape[0]**2)
        },
        "performance_data": results
    }

    with open(os.path.join(meta_dir, f"matrix_{matrix_id}.json"), 'w') as f:
        json.dump(meta_data, f, indent=2)


'''
This function will generate matrices

It will save the matrices as PNGs of a given dimension
Parameters: 
    - Sample amount
    - Matrix size
    _ Name of folder to save pngs

Each set of matrices for specified size will have a set of "allowable" block sizes
This will call a function that will return a list of acceptable block sizes based on the size

(i.e. a matrix of size 1000 could have blocks of size 2, 5, 10, 20, 50, 100, 125, 200, 250...
 We probably don't need all of those, so instead will will base it off percentages. i.e. 2%, 5%, 10%, 15%, 20%
 In reality matrices may be extremely large and have a larger range of block sizes but for now we will keep it simple)

5. Since we are resizing the PNG, I will change the store_pngs function by adding a parameter for width and height
    - If you do not wish to resize the image, simply pass the matrix size for width and height
'''
def generate_matrices(size, folder_name, sample_amount=3000):
    # list of blocks for all matrices of specified size
    best_size_array = []

    # variable to hold our matrices
    dataset = []

    # generate our list of acceptable blocks for current size
    acceptable_blocks = generate_acceptable_blocks(size)
    print(f"acceptable blocks for size {size}: {acceptable_blocks}")
    
    # samples per label
    block_samples = sample_amount // len(acceptable_blocks)
    # we want to generate sample_amount of matrices for each LABEL
    for block in acceptable_blocks:
        #print(f"CURRENT BLOCK IS: {block}. THERE SHOULD BE ONE SAMPLE PER BLOCK")
        for _ in range(block_samples):
            #print(f"Sample#: {j}")
            # list of blocks for our current matrix
            blocks = []

            # the number of blocks
            num_blocks = size // block
            #print(f"num_blocks is {num_blocks}")

            remainder = size % block
            #print(f"remainder is {remainder}")
            # add these blocks to list so we can generate matrix

            for k in range(num_blocks):
                blocks.append(block)
            if remainder != 0:
                blocks.append(remainder)
            #print(f"Block sizes are {blocks}")

            # generate the matrix
            A = generate_sparse(blocks, noise=0.01, de_noise=0.01)
            #print(f"A shape is {A.shape}")

            # store in list
            dataset.append(A)

            # find best block_jacobi size
            n = A.shape[0]
            #print(f"n is {n} and this size is {this_size}")
            b = np.ones(A.shape[0])
        
            best_size = find_best_block_size(n, A, b)
            best_size_array.append(best_size)

    store_pngs(dataset, best_size_array, 500, 500, base_dir=folder_name) # storing PNGS as 100x100 for now

    return


def generate_pipeline(size, folder_name, samples):
    candidates = FRACTION_CLASSES

    for i in range(samples):
        if i % 10 == 0: print(f"Generating sample {i}/{samples}...")

        # Jittered generation
        target_structure = random.uniform(0.05, 0.40)
        #target_structure = random.triangular(0.10, 0.40, 0.40)

        noise = random.uniform(0.001, 0.05)

        A = generate_jittered_matrix(size, target_structure, noise)
        b = np.ones(A.shape[0])

        results = solve_and_profile(A, b, candidates)
        save_data(A, results, i, folder_name, target_structure)


if __name__ == '__main__':
    args = parse_cli()
    #print(f"Arguments: {args.samples} samples of size {args.size}, folder name: {args.name}")
    #generate_matrices(args.size, args.name, sample_amount=args.samples)
    generate_pipeline(args.size, args.name, args.samples)
