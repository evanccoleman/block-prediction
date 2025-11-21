from scipy.sparse.linalg import gmres
from scipy.sparse import bmat, csc_matrix
from scipy.sparse.linalg import inv
import time
FRACTION_CLASSES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

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
    #if n % block_size != 0:
    #    raise ValueError("Matrix size must be divisible by block_size")

    inv_blocks = []
    for row_start in range(0, n, block_size):
        row_end = min(row_start + block_size, n)
        block = A[row_start:row_end, row_start:row_end]
        block = csc_matrix(block)
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


def find_best_block_size(n, A, b, eval_key='iterations'):
    pre_iters = {}
    counter = gmres_counter(disp=False)

    # Original run time with no preconditioner
<<<<<<< HEAD
    start_time = time.perf_counter() # rtol -> tol
=======
    start_time = time.perf_counter()
>>>>>>> 717b2a1c930a206908e8f5d072ffb10d038c02de
    x, i_exitCode = gmres(A, b, rtol=1e-2, callback=counter, maxiter=int(1e6))
    end_time = time.perf_counter()
    run_time = end_time - start_time
    #print(f"Original run time: {run_time}")
    #print(f"Original number of iterations: {counter.niter}")
    #print(f"rk is {counter.rk}")
    print(f"Exit code is {i_exitCode}")
    #plt.figure()
    #plt.plot(counter.residuals)
    #plt.show()
    divisors = [int(i * n) for i in FRACTION_CLASSES]
    for divisor in divisors:
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


<<<<<<< HEAD
        # rtol -> tol
        x_pre, exitCode = gmres(A, b, rtol=1e-2, callback=counter_pre,  M=M)
=======

        x_pre, exitCode = gmres(A, b, atol=1e-2, callback=counter_pre, maxiter=int(1e6), M=M)
>>>>>>> 717b2a1c930a206908e8f5d072ffb10d038c02de
        entire_end = time.perf_counter()
        end_after_M = time.perf_counter()


        M_run_time = end_time_for_M - start_time_for_M
        run_after_M = end_after_M - start_after_M
        entire_run = entire_end - entire_start
        #print(f"M run time: {M_run_time}")
        #print(f"Runtime after M: {run_after_M}")
        #print(f"Entire run time: {entire_run}")
        #print(f"Exit code: {exitCode}")
        #print(f"rk is {counter_pre.rk}")
        if exitCode == 0:
            pre_iters[divisor] = {
                'run_time': entire_run,
                'iterations': counter_pre.niter
                }

    # Check if any block sizes converged
    if not pre_iters:
        print("Warning: No block sizes converged. Using smallest block size as fallback.")
        # Return the smallest block size as a fallback
        if divisors:
            fallback_block = divisors[0]
            print(f"Fallback block size: {fallback_block}")
            return fallback_block
        else:
            print("Error: No divisors available. Returning None.")
            return None
    best_block = min(pre_iters, key=lambda k: pre_iters[k][eval_key])
    print("Made it through")
    print(f"Best block size: {best_block} with {pre_iters[best_block]['iterations']} iterations and {pre_iters[best_block]['run_time']} run time.")
    print(pre_iters)
    return best_block

