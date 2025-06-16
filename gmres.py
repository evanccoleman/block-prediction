import ssgetpy
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres
from scipy.sparse import diags, bmat, csr_matrix
from scipy.sparse.linalg import inv

def generate_5_point_stencil_matrix(n):
    """Generates the matrix for a 5-point stencil on an n x n grid.

    Args:
        n: The size of the grid (n x n).

    Returns:
        A sparse matrix representing the 5-point stencil.
    """

    main_diag = -4 * np.ones(n * n)
    off_diag = np.ones(n * n - 1)
    off_diag[np.arange(1, n * n) % n == 0] = 0  # Correct for edges
    super_diag = np.ones(n * n - n)
    sub_diag = np.ones(n * n - n)

    diagonals = [main_diag, off_diag, off_diag, super_diag, sub_diag]
    offsets = [0, -1, 1, -n, n]

    laplacian_matrix = diags(diagonals, offsets, shape=(n * n, n * n), format="csr")
    return laplacian_matrix

def block_jacobi_preconditioner(A, block_size):
    """
    Creates a block Jacobi preconditioner for a sparse matrix A.

    Args:
        A (scipy.sparse.csr_matrix): The input sparse matrix.
        block_size (int): The size of the blocks.

    Returns:
        scipy.sparse.csr_matrix: The block Jacobi preconditioner matrix.
    """
    n = A.shape[0]
    num_blocks = n // block_size
    if n % block_size != 0:
        raise ValueError("Matrix size must be divisible by block_size")

    inv_blocks = []
    for i in range(num_blocks):
        row_start = i * block_size
        row_end = row_start + block_size
        block = A[row_start:row_end, row_start:row_end]
        inv_blocks.append(inv(block))

    # Create the block diagonal preconditioner matrix
    M = bmat([[inv_blocks[i] if i == j else None for j in range(num_blocks)] for i in range(num_blocks)], format="csr")
    return M

class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

# Ax = b
#   A comes from physics
#   x is a guess that starts at 0's (probably)
#   b defines the edges of the problem
n = 50
A = generate_5_point_stencil_matrix(n)
b = np.ones([n*n, 1])
#print(A)
plt.spy(A)
plt.show()
M = block_jacobi_preconditioner(A, 100)
plt.figure()
plt.spy(M)
plt.show()
counter = gmres_counter()
counter_pre = gmres_counter()
x = gmres(A, b, rtol=1e-2,callback=counter)
print("="*60)
x_pre = gmres(A, b, rtol=1e-2,callback=counter_pre,M=M)
print(x)

print(b)


