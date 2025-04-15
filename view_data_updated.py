import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import h5py
import time

handle = h5py.File('./synthetic_data.h5', 'r+')
data = np.array(handle['matrix_of_hard_64'])
labels = np.array(handle['labels_for_hard_64'])

handle.close()
for i in range(data.shape[2]):
    matplotlib.pyplot.spy(data[:,:,i])
    matplotlib.pyplot.show()
    time.sleep(0.5)
block_size = labels[0]
print(f"Block size is {block_size}")
matrix_size = data.shape[0]
print(f"Matrix is size {matrix_size}")