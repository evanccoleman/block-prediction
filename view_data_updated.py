import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import h5py

handle = h5py.File('./synthetic_data.h5', 'r+')
data = np.array(handle['matrix_of_100'])
labels = np.array(handle['labels_for_100'])

handle.close()
matplotlib.pyplot.spy(data[:,:,128])
matplotlib.pyplot.show()
block_size = labels[0]
print(f"Block size is {block_size}")
matrix_size = data.shape[0]
print(f"Matrix is size {matrix_size}")