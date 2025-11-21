import os
import random
import matplotlib
import numpy as np
import scipy.sparse as sp
from PIL import Image
from matplotlib import pyplot as plt
from block_jacobi import find_best_block_size
import argparse

FRACTION_CLASSES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
#print(f"Fraction classes: {FRACTION_CLASSES}")
random.seed(42)
np.random.seed(42)

def parse_cli():
    parser = argparse.ArgumentParser(description="PNG Builder")
    parser.add_argument("--size", type=int, default=500, help="Size of matrices to generate")
    parser.add_argument("--name", "-n", type=str, help="Folder name to save to")
    parser.add_argument("--samples", "-s", type=int, default=3000, help="The number of samples")
    return parser.parse_args()

def generate_sparse(block_sizes, noise=0.1, de_noise=0.1, random_state=42):
    rng = np.random.default_rng(random_state)
    blocks = []
    for block in block_sizes:
        blocks.append(
            sp.random(
                block,
                block,
                density=1.0,
                format="coo",
                data_rvs=lambda k: rng.uniform(-0.1, 0.0, size=k),
            )
        )
    A = sp.block_diag(blocks, format='coo')
    n = A.shape[0]

    if noise > 0.0 and A.nnz:
        noise_mask = sp.random(
            n,
            n,
            density=noise,
            format="coo",
            data_rvs=lambda k: rng.uniform(-.1, 0.0, size=k),
        )
        A = A + noise_mask
        A.sum_duplicates()
    if de_noise > 0.0 and A.nnz:
        A = A.tocoo()
        # keep mask is boolean array, drops entires based on de_noise parameter
        keep_mask = rng.random(A.nnz) >= de_noise
        A = sp.coo_matrix(
            (A.data[keep_mask],
             (A.row[keep_mask], A.col[keep_mask])),
            shape=A.shape)
    A.setdiag(100)
    return A.tocsr()

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
    matrix_size = data[0].shape[0]
    classes = [int(i * matrix_size) for i in FRACTION_CLASSES]
    #size_folder = os.path.join(base_dir, f'size_{matrix_size}')
    #os.makedirs(size_folder, exist_ok=True)
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
        for j in range(block_samples):
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

if __name__ == '__main__':
    args = parse_cli()
    #print(f"Arguments: {args.samples} samples of size {args.size}, folder name: {args.name}")
    generate_matrices(args.size, args.name, sample_amount=args.samples)
