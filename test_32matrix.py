import numpy as np
from scipy.stats import chi2

def get_binary_rank_32(rows):
    """
    Computes the rank of a 32x32 binary matrix over GF(2).
    Uses bitwise Gaussian elimination.
    """
    rank = 0
    # Create a local copy of the rows to manipulate
    matrix = np.array(rows, dtype=np.uint32)
    
    for i in range(32):
        # We look for a pivot in the i-th bit position (from left to right)
        pivot_mask = 1 << (31 - i)
        
        # Find a row from current 'rank' downwards that has a 1 in this position
        pivot_row = -1
        for r in range(rank, 32):
            if matrix[r] & pivot_mask:
                pivot_row = r
                break
        
        if pivot_row != -1:
            # Swap current row with pivot row
            matrix[rank], matrix[pivot_row] = matrix[pivot_row], matrix[rank]
            
            # Eliminate this bit from all other rows using XOR
            for r in range(32):
                if r != rank and (matrix[r] & pivot_mask):
                    matrix[r] ^= matrix[rank]
            
            rank += 1
            
    return rank

def matrix_rank_32x32_test(rng_func, num_matrices=40000):
    """
    Production-level 32x32 Matrix Rank Test.
    
    Args:
        rng_func: A generator function that yields an array of 32-bit uints.
        num_matrices: Number of trials (Diehard uses 40,000).
    """
    # Theoretical probabilities for Rank 32, 31, and <=30
    p = np.array([0.2887880952, 0.5775761905, 0.1336357143])
    counts = np.zeros(3)

    print(f"Executing 32x32 Matrix Rank Test: {num_matrices} trials...")

    for _ in range(num_matrices):
        # 1. Select 32 integers to form the rows of the matrix
        matrix_rows = rng_func(32)
        
        # 2. Calculate the rank
        r = get_binary_rank_32(matrix_rows)
        
        # 3. Bin the results
        if r == 32:
            counts[0] += 1
        elif r == 31:
            counts[1] += 1
        else:
            counts[2] += 1

    # 4. Chi-Squared Goodness of Fit Test
    expected = p * num_matrices
    # Chi-square statistic: sum((observed - expected)^2 / expected)
    chi_stat = np.sum((counts - expected)**2 / expected)
    
    # Degrees of freedom = (Number of bins - 1) = 2
    p_value = 1 - chi2.cdf(chi_stat, df=2)

    print("-" * 30)
    print(f"Rank 32 (Expected: {int(expected[0])}): {int(counts[0])}")
    print(f"Rank 31 (Expected: {int(expected[1])}): {int(counts[1])}")
    print(f"Rank <=30 (Expected: {int(expected[2])}): {int(counts[2])}")
    print(f"P-Value: {p_value:.6f}")

    if 0.0001 < p_value < 0.9999:
        print("TEST RESULT: PASS")
    else:
        print("TEST RESULT: FAIL")
    
    return p_value

# Usage Example:
def secure_rng_source(n):
    # Using numpy's MT19937 or PCG64
    return np.random.randint(0, 0xFFFFFFFF, size=n, dtype=np.uint32)

matrix_rank_32x32_test(secure_rng_source)