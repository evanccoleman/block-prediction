import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import h5py


def create_matrix(block_sizes, diagonal_noise=10, matrix_noise=10, matrix_size=100, zero_prob=.20):
    matrix = np.zeros((matrix_size, matrix_size))
    matplotlib.pyplot.spy(matrix)

    # Place blocks on diagonal
    curr_index = 0
    for size in block_sizes:
        if curr_index + size > matrix_size:
            break # do not let blocks exceed matrix size
        # create block with added noise
        block = np.random.randint(1, 10, (size, size))
        zero_mask = np.random.rand(size, size) < zero_prob
        block[zero_mask] = 0

        # Place the block in the matrix
        matrix[curr_index:curr_index + size, curr_index:curr_index + size] = block
        curr_index += size + 2  # Leave space between blocks
    # Add noise to diagonal
    # NEED TO CHANGE THIS TO ADD NOISE TO BLOCKS
    for n in range(diagonal_noise):
        i = np.random.randint(0, matrix_size)
        j = np.clip(i + np.random.randint(-3, 4), 0, matrix_size - 1)  # Keep the index near diagonal
        matrix[i, j] = np.random.randint(1, 10)
    # Add noise to entire thing
    # need to adjust because it's hard to add a lot of noise
    for n in range(matrix_noise):
        i = np.random.randint(0, matrix_size)
        j = np.random.randint(0, matrix_size)
        matrix[i, j] = np.random.randint(1000, 10000)
    return matrix

def generate_block_sizes(matrix_size, min_block_size=2, max_block_size=100):
    block_sizes = []
    space_left = matrix_size
    while space_left > min_block_size:
        block_size = np.random.randint(min_block_size, min(max_block_size, space_left) + 1)
        block_sizes.append(block_size)
        space_left -= block_size
    return block_sizes

# GENERATE THE MATRIX
size = 1000
max_block = size/5

# Get block sizes to ensure blocks stretch across the diagonal
block_sizes = generate_block_sizes(size, min_block_size=2, max_block_size=max_block)
#print(block_sizes)


matrix = create_matrix(block_sizes, diagonal_noise=3000, matrix_noise=100000, matrix_size=1000, zero_prob=.20)
#print(matrix)
matplotlib.pyplot.spy(matrix)
matplotlib.pyplot.show()

#matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

#matplotlib.pyplot.spy(matrix)
#matplotlib.pyplot.show()

#diagonal_elements = np.array([1,2,3,4])
#diagonal_matrix = np.diag(diagonal_elements)
#matplotlib.pyplot.spy(diagonal_matrix)
#matplotlib.pyplot.show()
