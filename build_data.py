import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import h5py

# function takes in matrix size, range of block sizes, noise, and de-noise
def generate_data(matrix_size, block_sizes, noise=0.2, de_noise=0.4):

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
    # these values should be from 0-0.5
    matrix[noise_mask] = 1

    return matrix

# generate multiple matrices of a specific size, giving uniform block sizes
def generate_multiple_uniform(matrix_size, amount=1000):
    # we will go by a general rule that max noise=0.3 and max denoise=0.4
    # in order to still have "visible" blocks

    # uniform blocks:
    blocks = generate_fixed_blocks(matrix_size)

    block_num = len(blocks)
    blocks_array = np.empty((block_num, amount))
    array_of_matrices = np.empty((matrix_size, matrix_size, amount))
    for i in range(amount):
        array_of_matrices[:, :, i] = generate_data(matrix_size, blocks, noise=np.random.uniform(0, 0.4),
                                        de_noise=np.random.uniform(0, 0.5))
        blocks_array[:,i] = blocks

    #blocks_array = np.array(blocks)
    #blocks_array = np.expand_dims(blocks_array, axis=-1)

    # SAVE THE NEW MATRIX TO AN H5 FILE
    with h5py.File('synthetic_data.h5', 'a') as f:

        matrixset_name = 'matrix_of_' + str(matrix_size)
        f.create_dataset(matrixset_name, data=array_of_matrices)

        labelset_name = 'labels_for_' + str(matrix_size)
        f.create_dataset(labelset_name, data=blocks_array)

    print("Matrices of size " + str(matrix_size) + " saved to 'synthetic_data.h5'")
    print("Block sizes of matrices: " + str(matrix_size))
    return array_of_matrices


# this will generate a list of block sizes
# gives you control over min and max block sizes
def generate_block_sizes(matrix_size, block_size_range=(0.05, 0.2)):

    min_block = int(matrix_size * block_size_range[0])
    max_block = int(matrix_size * block_size_range[1])
    blocks = []
    total_size = 0
    block_size = 0

    while total_size < matrix_size:
        block_size = np.random.randint(min_block, max_block)
        if total_size + block_size > matrix_size:
            block_size = matrix_size - total_size
        blocks.append(block_size)
        total_size += block_size

    return blocks # our list of block sizes

def generate_fixed_blocks(matrix_size, block_ratio=0.05):
    block_size = int(matrix_size * block_ratio)
    blocks = []
    num_blocks = matrix_size // block_size
    for i in range(num_blocks):
        blocks.append(block_size)
    return blocks

# def save_to_h5(matrix, file_name='synthetic_data.h5', dataset_name):
#     # SAVE THE NEW MATRIX TO AN H5 FILE
#     with h5py.File(file_name, 'a') as f:
#
#         matrixset_name = dataset_name
#         f.create_dataset(matrixset_name, data=matrix)
#
#         labelset_name = 'labels' + str(num_matrix)
#         f.create_dataset(labelset_name, data=block_sizes)
#     print("Matrix saved to 'synthetic_data.h5'")


# test out matrix
size = 1000
#block_sizes = generate_fixed_blocks(size)
#print(block_sizes)

matrices = generate_multiple_uniform(size)
handle = h5py.File('./synthetic_data.h5', 'r+')
#data = np.array(handle['matrix_of_600'])


handle.close()
#matplotlib.pyplot.spy(data[:,:,100])
#matplotlib.pyplot.show()


