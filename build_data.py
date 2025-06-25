import random

import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import h5py
from PIL import Image
import tensorflow as tf
import io
import os
import math

'''
Function to generate matrices.
Parameters: 
    - matrix_size: The size of the matrix.
    - block_sizes: The list of block sizes.
    - noise: How much noise you would like to add.
    - de_noise: How much noise to remove, or "denoise"
Function returns the generated matrix.
'''
def generate_data(matrix_size, block_sizes, noise=0.2, de_noise=0.4):

    # Initialize a numpy array of all zeros
    matrix = np.zeros((matrix_size, matrix_size))

    # In order to add/remove noise later, and avoid the blocks, we need to create a mask
    block_mask = np.zeros_like(matrix, dtype=bool)

    # Starting position
    row, col = 0, 0

    # First block: set the entries in the blocks along the diagonal to 1
    for block in block_sizes:
        if row + block > matrix_size or col + block > matrix_size:
            break # don't let block exceed matrix boundaries

        # Fill block with ones
        matrix[row:row+block, col:col+block] = np.random.uniform(low=-1.0, high=0.0, size=(block,block))
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

    # Apply noise to matrix
    # Should be negative as well
    num_noise_cells = noise_mask.sum()
    if num_noise_cells:
        matrix[noise_mask] = np.random.uniform(-1.0, 0.0, size=num_noise_cells)
    
    # Reset rows and columns to ensure diagonal is all 1s
    row, col = 0, 0

    for i in range(matrix_size + 1):
        if row + 1> matrix_size or col + 1 > matrix_size:
            break
        matrix[row:row + 1, col:col + 1] = 1

        # Move diagonally
        row += 1
        col += 1

    return matrix

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
    for i in range(amount):
        # Random block ratio
        block_ratio = np.random.uniform(0.05, 0.1)
        #print(block_ratio)

        # Generate block sizes for current matrix
        blocks, block_size = generate_fixed_blocks(matrix_size, block_ratio=block_ratio)
        block_size_array.append(block_size)
        print("Genreated block sizes")
        # Generate the matrix
        array_of_matrices[:, :, i] = generate_data(matrix_size, blocks, noise=0.2, de_noise=0.2)
        print("Generated matrices")
    #blocks_array = np.array(blocks)
    #blocks_array = np.expand_dims(blocks_array, axis=-1)



    # SAVE THE NEW MATRIX TO AN H5 FILE
    with h5py.File('synthetic_data.h5', 'w') as f:

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


# this will generate a list of block sizes
# gives you control over min and max block sizes
def generate_block_sizes(matrix_size, block_size_range=(0.05, 0.07)):

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

'''
Function generates a list of fixed block sizes.
Parameters:
    - matrix_size: The size of the matrix.
    - block_ratio: Ratio of block to matrix, i.e. what percentage of matrix size will our blocks be
Function returns the list of blocks and the size.
'''
def generate_fixed_blocks(matrix_size, block_ratio=0.07):
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
Function converts matrix to png format
Parameters:
    - matrix: The matrix to convert.
Returns the PIL image object.
'''
def matrix_to_png(matrix):
    # 255 is white, 0 is black.
    # when we use matplotlib to show matrices, we view the 1s as black so we need to invert matrix
    matrix = (1- matrix) * 255
    matrix = matrix.astype(np.uint8)
    img = Image.fromarray(matrix)
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
def store_pngs(data, labels, width, height, base_dir='png_with_noise'):
    os.makedirs(base_dir, exist_ok=True)
    matrix_size = data[:, :, 0].shape[0]
    size_folder = os.path.join(base_dir, f'size_{matrix_size}')
    os.makedirs(size_folder, exist_ok=True)
    for label in labels:
        label_folder = os.path.join(size_folder, f'label_{label}')
        os.makedirs(label_folder, exist_ok=True)
    for i in range(data.shape[2]):
        matrix = data[:,:,i]
        img = matrix_to_png(matrix)
        if (width != matrix_size) or (height != matrix_size):
            img = img.resize((width, height))
        file_path = os.path.join(size_folder, f'label_{labels[i]}',f'matrix_{i}.png')
        img.save(file_path)

def generate_acceptable_blocks(matrix_size):

    block_list = []
    grouped_list = []
    # all divisors of matrix size
    divisors = [d for d in range(1, matrix_size + 1) if matrix_size % d == 0]
    
    # taking all divisors greater than 2% but less than 30% of matrix size
    for divisor in divisors:
        if (divisor) < (matrix_size * .30) and (divisor) > (matrix_size * .02):
            block_list.append(divisor)
    if len(block_list) > 4:
        group_size = len(block_list) // 4
        mid_group = group_size // 2
        start = 0
        for i in range(4):
            end = start + group_size
            grouped_list.append(block_list[start:end])
            start = end
        ideal_list = [group[mid_group] for group in grouped_list] # I should technically have an if/else case for empty groups, but large matrices wont allow for empty group
        return ideal_list
    else:
        return block_list
'''
This function will generate varying matrix sizes

It will save the matrices as PNGs of a given dimension
Parameters: 
    - Number of matrix sizes you want to generate
    - Range of matrix sizes allowed
    - Sample amount for each matrix size (i.e. 1000 samples of size 64)
    
Each set of matrices for specified size will have a set of "allowable" block sizes
This will call a function that will return a list of acceptable block sizes based on the size

(i.e. a matrix of size 1000 could have blocks of size 2, 5, 10, 20, 50, 100, 125, 200, 250...
 We probably don't need all of those, so instead will will base it off percentages. i.e. 2%, 5%, 10%, 15%, 20%
 In reality matrices may be extremely large and have a larger range of block sizes but for now we will keep it simple)
 
1. Generate list of matrix sizes
2. Generate list of acceptable block sizes for said matrix size
3. For x amount of times (specified sample amount), a random choice from the list of acceptable block sizes is chosen
4. Calls generate_data and generates an array of matrices with that specified block size
5. Since we are resizing the PNG, I will change the store_pngs function by adding a parameter for width and height
    - If you do not wish to resize the image, simply pass the matrix size for width and height
'''
def generate_varying_matrices(size_amount, sample_amount=1000, size_range=(100,501)):
    used_sizes = []
    # we want size_amount of matrix sizes
    for i in range(size_amount):
        # pick the current size randomly, but I need it to be divisible by 100 for ease
        this_size = random.choice(range(size_range[0], size_range[1], 100))
        while this_size in used_sizes:
            this_size = random.choice(size_range)
        used_sizes.append(this_size)
        print(f"Current size is: {this_size}")

        # list of blocks for all matrices of specified size
        block_size_array = []

        # variable to hold our matrices
        array_of_matrices = np.empty((this_size, this_size, sample_amount))

        # generate our list of acceptable blocks for current size
        acceptable_blocks = generate_acceptable_blocks(this_size)

        # we want to generate sample_amount of matrices for each size
        for j in range(sample_amount):
            # list of blocks for our current matrix
            blocks = []

            # randomly pick a size from our list of acceptable blocks
            block_size = random.choice(acceptable_blocks)

            # the number of blocks
            num_blocks = this_size // block_size

            # add these blocks to list so we can generate matrix
            for k in range(num_blocks):
                blocks.append(block_size)

            # append block size to list of all blocks for all matrices of this size
            block_size_array.append(block_size)
            # generate the matrix
            array_of_matrices[:, :, j] = generate_data(this_size, blocks, noise=0.20, de_noise=0.01)
            if (j == 0):
                print(array_of_matrices[:,:,j])

        data = np.array(array_of_matrices)
        store_pngs(data, block_size_array, this_size, this_size)

        #print(pngs[0].size)
        #pngs[0].show()
        # compare to matplotlib image
        matplotlib.pyplot.spy(data[:,:,0])
        matplotlib.pyplot.show()
    return

#print(f"Acceptable block sizes for matrix of size 10000: {generate_acceptable_blocks(10000)}")

#generate_varying_matrices(1, sample_amount=1000, size_range=(128,128))
generate_multiple_uniform(128)




# Resizing PNGS:
# image = Image.open("path/to/your/image.jpg")
# new_width = 500
# new_height = 300
# resized_image = image.resize((new_width, new_height))

#***************************************************** HARD CODED SECTION *****************************************************#
# hard_size=64
# hard_amount=5000
# array_of_matrices = np.empty((hard_size, hard_size, hard_amount))
# block_size_array = []
# hard_list = [2, 4, 8, 16]
#
# for i in range(hard_amount):
#     # list of blocks for current matrix
#     blocks=[]
#
#     # block size for current matrix
#     block_size = random.choice(hard_list)
#
#     # number of blocks
#     num_blocks = hard_size // block_size
#     # add these blocks to list so we can generate matrix
#     for j in range(num_blocks): # NOTE: Error was here
#         blocks.append(block_size)
#     # append block size to list of all blocks for all matrices
#     block_size_array.append(block_size)
#
#     # generate the matrix
#     array_of_matrices[:, :, i] = generate_data(hard_size, blocks, noise=0.01, de_noise=0.01)
#     if (i == 0):
#         print(array_of_matrices[:,:,i])
#
#
#
# data = np.array(array_of_matrices)
# store_pngs(data, block_size_array)
#
# #print(pngs[0].size)
# #pngs[0].show()
# # compare to matplotlib image
# matplotlib.pyplot.spy(data[:,:,0])
# matplotlib.pyplot.show()

