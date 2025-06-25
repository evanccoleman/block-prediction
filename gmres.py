import ssgetpy
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import gmres
from scipy.sparse import diags, bmat, csr_matrix
from scipy.sparse.linalg import inv
import time
from scipy.io import mmread
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
    #num_blocks = n // block_size
    #if n % block_size != 0:
    #    raise ValueError("Matrix size must be divisible by block_size")

    inv_blocks = []
    for row_start in range(0, n, block_size):
        row_end = min(row_start + block_size, n)
        block = A[row_start:row_end, row_start:row_end].tocsc()
        inv_blocks.append(inv(block))

    # Create the block diagonal preconditioner matrix
    num_blocks = len(inv_blocks)
    M = bmat([[inv_blocks[i] if i == j else None for j in range(num_blocks)] for i in range(num_blocks)], format="csr")
    return M

class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.rk = 0
        self.residuals = []
    def __call__(self, rk=None):
        self.niter += 1
        self.rk = rk
        self.residuals.append(self.rk)
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

def find_best_block_size(n, A, b):
    eval = 'run_time'
    #eval = 'iterations'
    pre_iters = {}
    counter = gmres_counter(disp=False)

    # Original run time with no preconditioner
    start_time = time.perf_counter()
    x, i_exitCode = gmres(A, b, rtol=1e-2, callback=counter, maxiter=int(1e6))
    end_time = time.perf_counter()
    run_time = end_time - start_time
    print(f"Original run time: {run_time}")
    print(f"Original number of iterations: {counter.niter}")
    print(f"rk is {counter.rk}")
    print(f"Exit code is {i_exitCode}")
    plt.figure()
    plt.plot(counter.residuals)
    plt.show()

    for divisor in range(1, n):
    #for divisor in divisors:
        print(f"Testing block size: {divisor}")
        counter_pre = gmres_counter(disp=False)
        # track time for M
        start_time_for_M = time.perf_counter()
        # track entire runtime
        entire_start = time.perf_counter()
        M = block_jacobi_preconditioner(A, divisor)
        # M run time end
        end_time_for_M = time.perf_counter()
        # track time after M
        start_after_M = time.perf_counter()



        x_pre, exitCode = gmres(A, b, rtol=1e-2, callback=counter_pre,  M=M)
        entire_end = time.perf_counter()
        end_after_M = time.perf_counter()


        M_run_time = end_time_for_M - start_time_for_M
        run_after_M = end_after_M - start_after_M
        entire_run = entire_end - entire_start
        print(f"M run time: {M_run_time}")
        print(f"Runtime after M: {run_after_M}")
        print(f"Entire run time: {entire_run}")
        print(f"Exit code: {exitCode}")
        print(f"rk is {counter_pre.rk}")
        if exitCode == 0:
            pre_iters[divisor] = {
                'run_time': entire_run,
                'iterations': counter_pre.niter
                }


    best_block = min(pre_iters, key=lambda k: pre_iters[k][eval])
    print("Made it through")
    print(f"Best block size: {best_block} with {pre_iters[best_block]['iterations']} iterations and {pre_iters[best_block]['run_time']} run time.")
    print(pre_iters)
    return best_block




if __name__ == '__main__':
    # Ax = b
    #   A comes from physics
    #   x is a guess that starts at 0's (probably)
    #   b defines the edges of the problem

    #n = 100
    #A = generate_5_point_stencil_matrix(n)
    A = mmread("suitesparse/bcsstk01.mtx").tocsr() # online matrix
    n = A.shape[0]
    print(n)
    print(A.shape)

    b = np.ones(A.shape[0])
    best_block_size = find_best_block_size(n, A, b)
    print("=" * 100)
    #print(A)
    plt.spy(A, markersize=1)
    plt.show()
    # grab all divisors

        # M = block_jacobi_preconditioner(A, 5)
    # plt.figure()
    # plt.spy(M, markersize=1)
    # plt.show()
    #
    # counter = gmres_counter(disp=False)
    # counter_pre = gmres_counter(disp=False)
    #
    # start_time = time.perf_counter()
    # x = gmres(A, b, rtol=1e-2,callback=counter)
    # end_time = time.perf_counter()
    # run_time = end_time - start_time
    # print(f"Original run time: {run_time}")
    # print(f"Original number of iterations: {counter.niter}")
    # print("="*60)
    #
    # pre_start_time = time.perf_counter()
    # x_pre = gmres(A, b, rtol=1e-2,callback=counter_pre,M=M)
    # pre_end_time = time.perf_counter()
    # pre_run_time = pre_end_time - pre_start_time
    #
    #
    # print(f"Preconditioner run time: {pre_run_time}")
    # print(f"Preconditioner number of iterations: {counter_pre.niter}")
    # #print(x)
    #
    # #print(b)
