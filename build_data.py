import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import h5py

# function takes in matrix size, range of block sizes, noise, and de-noise
def generate_data(matrix_size, block_sizes, noise=0.2, de_noise=0.2):

    # initialize a numpy array of all zeros
    matrix = np.zeros((matrix_size, matrix_size))

    # in order to add/remove noise later, and avoid the blocks, we need to create a mask
    block_mask = np.zeros_like(matrix, dtype=bool)

    # starting pos
    row, col = 0, 0

    # first block: set the entries in the blocks along the diagonal to 1
    for block in block_sizes:
        if row + block > matrix_size or col + block > matrix_size:
            break # don't let block exceed matrix boundaries

        # fill block with ones
        matrix[row:row+block, col:col+block] = 1
        block_mask[row:row+block, col:col+block] = True # also add to block_mask

        # move diagonally
        row += block
        col += block

    # second block: look only at entries that are 1 and flip them to a 0 with a probability
    # related to the "de_noise" parameter
    denoise_mask = np.random.rand(*matrix.shape) < de_noise # Creates a boolean mask
    matrix[denoise_mask] = 0 # Sets these "de_noise" positions to a 0

    # third block: look only at the entries that are 0 (preferably not in the original block space)
    # and flip them to a 1 with probability related to the "noise" parameter
    noise_mask = np.random.rand(*matrix.shape) < noise

    # Combine noise_mask and block_mask to ensure we avoid the block positions
    noise_mask = np.logical_and(noise_mask, ~block_mask)

    # apply noise to matrix
    matrix[noise_mask] = 1

    # SAVE THE NEW MATRIX TO AN H5 FILE
    with h5py.File('generated_matrix.h5', 'w') as f:
        f.create_dataset('new_matrix', data=matrix)
    print("Matrix saved to 'generated_matrix.h5'")
    return matrix

# this will generate a list of block sizes
# gives you control over min and max block sizes
def generate_block_sizes(matrix_size, block_size_range=(0.05, 0.2)):
    min_block = int(matrix_size * block_size_range[0])
    max_block = int(matrix_size * block_size_range[1])
    blocks = []
    total_size = 0

    while total_size < matrix_size:
        block_size = np.random.randint(min_block, max_block)
        if total_size + block_size > matrix_size:
            block_size = matrix_size - total_size
        blocks.append(block_size)
        total_size += block_size

    return blocks # our list of block sizes

# test out matrix
size = 100
block_sizes = generate_block_sizes(size)

matrix = generate_data(size, block_sizes)
matplotlib.pyplot.spy(matrix)
matplotlib.pyplot.show()


