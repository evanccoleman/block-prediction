import ssgetpy
from numpy.core.defchararray import lower
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres
from scipy.sparse import diags, bmat, csc_matrix, csr_matrix, block_diag, lil_matrix
from scipy.sparse.linalg import inv, LinearOperator
import time
from scipy.io import mmread
import random

import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import h5py
from PIL import Image
#import tensorflow as tf
import io
import os
import math
from block_jacobi import block_jacobi_preconditioner, gmres_counter, find_best_block_size

def generate_data(matrix_size, block_sizes, noise=0.2, de_noise=0.4):
    # Initialize a numpy array of all zeros
    matrix = np.zeros((matrix_size, matrix_size))

    # In order to add/remove noise later, and avoid the blocks, we need to create a mask
    block_mask = np.zeros_like(matrix, dtype=bool)

    # Starting position
    row, col = 0, 0

    # First block: set the entries in the blocks along the diagonal to non-zeros
    for block in block_sizes:
        if row + block > matrix_size or col + block > matrix_size:
            break # don't let block exceed matrix boundaries

        # Fill block with NOT ones
        matrix[row:row+block, col:col+block] = np.random.uniform(
        -1.0, 0.0, size=(block, block)
    )
        block_mask[row:row+block, col:col+block] = True # also add to block_mask

        # Move diagonally
        row += block
        col += block

    # Second block: look only at entries that are 1 and flip them to a 0 with a probability
    # related to the "de_noise" parameter
    denoise_mask = np.random.rand(*matrix.shape) < de_noise # Creates a boolean mask
    matrix[denoise_mask] = 0 # Sets these "de_noise" positions to a 0

    # Third block: look only at the entries that are 0 (preferably not in the original block space)
    # and flip them to a 1 with probability related to the "noise" parameter
    noise_mask = np.random.rand(*matrix.shape) < noise

    # Combine noise_mask and block_mask to ensure we avoid the block positions
    noise_mask = np.logical_and(noise_mask, ~block_mask)

    num_noise_cells = noise_mask.sum()
    if num_noise_cells:  # skip if mask is empty
        matrix[noise_mask] = np.random.uniform(-1.0, 0.0, size=num_noise_cells)

    # reset and ensure diagonal is completely ones
    row, col = 0, 0

    for i in range(matrix_size + 1):
        if row + 1 > matrix_size or col + 1 > matrix_size:
            break
        matrix[row:row + 1, col:col + 1] = 1

        # Move diagonally
        row += 1
        col += 1
    return matrix

'''
Function generates a list of fixed block sizes.
Parameters:
    - matrix_size: The size of the matrix.
    - block_ratio: Ratio of block to matrix, i.e. what percentage of matrix size will our blocks be
Function returns the list of blocks and the size.
'''
def generate_fixed_blocks(matrix_size, block_ratio=0.05):
    block_size = int(matrix_size * block_ratio)
    #print(f"The uniform block size is {block_size}")
    blocks = []
    #print(f"The matrix size is {matrix_size}")
    num_blocks = matrix_size // block_size
    #print(f"The matrix size divided by block size is is {num_blocks}")
    for i in range(num_blocks):
        blocks.append(block_size)
    return blocks, block_size

'''
Function will generate multiple matrices of a specified size.
Parameters:
    - matrix_size: The size of the matrix.
    - amount: The amount of matrices to generate. Our sample size.
Function will save the matrices to h5 file. 
Returns an array of matrices.
'''
def generate_multiple_uniform(matrix_size, amount=1000):
    # We will go by a general rule that max noise=0.3 and max denoise=0.4
    # in order to still have "visible" blocks
    array_of_matrices = np.empty((matrix_size, matrix_size, amount))
    block_size_array = []
    fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    for i in range(amount):
        # Random block ratio
        block_ratio = random.choice(fractions)
        #print(block_ratio)

        # Generate block sizes for current matrix
        blocks, block_size = generate_fixed_blocks(matrix_size, block_ratio=block_ratio)


        # Generate the matrix
        A = generate_data(matrix_size, blocks, noise=0.0, de_noise=0.0)
        array_of_matrices[:, :, i] = A
        n = A.shape[0]
        b = np.ones(A.shape[0])
        best_block_size = find_best_block_size(n, A, b)
        block_size_array.append(best_block_size)

    #blocks_array = np.array(blocks)
    #blocks_array = np.expand_dims(blocks_array, axis=-1)



    # SAVE THE NEW MATRIX TO AN H5 FILE
    with h5py.File('tested_synthetic_class.h5', 'w') as f:

        matrixset_name = 'matrix_of_' + str(matrix_size)
        f.create_dataset(matrixset_name, data=array_of_matrices)


        labelset_name = 'labels_for_' + str(matrix_size)
        f.create_dataset(labelset_name, data=block_size_array)


        data = np.array(f[matrixset_name])
        labels = np.array(f[labelset_name])
        #print(data.shape)
        #print(labels.shape)


    print("Matrices of size " + str(matrix_size) + " saved to 'synthetic_data.h5'")
    #print("Block sizes of matrices: " + str(matrix_size))
    return array_of_matrices

matrices_array = generate_multiple_uniform(128, amount=3000)

