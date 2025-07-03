import matplotlib.pyplot
import numpy as np
import h5py

DATA_DATASET = 'matrixset'
DIAGONALS_DATASET = 'diagonalset'
WINDOW = 10

handle = h5py.File('./artificial.h5', 'r+')
# load the data and move the sample axis to the front
data = np.moveaxis(np.array(handle[DATA_DATASET]), -1, 0)

width = data.shape[1]
samples = data.shape[0]
diagonalset = np.zeros((samples, 2 * WINDOW + 1, width), dtype=np.float32)
handle.close()

matplotlib.pyplot.spy(data[0])
matplotlib.pyplot.show()