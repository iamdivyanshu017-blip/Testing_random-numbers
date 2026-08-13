import numpy as np
from scipy.stats import chi2

# NIST Constants for 32x32 Binary Matrices
M = 32
Q = 32
# Source: NIST SP 800-22 Rev 1a, Section 2.5.4
RANK_PROBABILITIES = np.array([0.2888, 0.5776, 0.1336])
DEGREES_OF_FREEDOM = 2

def _binary_matrix_rank(matrix: np.ndarray) -> int:
    """
    Calculates the rank of a binary matrix over GF(2) using Gaussian elimination.
    """
    A = matrix.copy()
    num_rows, num_cols = A.shape
    
    rank = 0
    pivot_row = 0
    
    for col in range(num_cols):
        if pivot_row >= num_rows:
            break
            
        # Find a pivot in the current column
        i = pivot_row
        while i < num_rows and A[i, col] == 0:
            i += 1
            
        # If a 1 is found
        if i < num_rows:
            # Swap rows if necessary
            if i != pivot_row:
                A[[i, pivot_row]] = A[[pivot_row, i]]
            
            # Eliminate other rows
            for j in range(num_rows):
                if j != pivot_row and A[j, col] == 1:
                    # Row j = Row j XOR Pivot Row
                    A[j, :] ^= A[pivot_row, :]
            
            rank += 1
            pivot_row += 1
            
    return rank

def matrix_rank_test(bit_array: np.ndarray, alpha: float = 0.01) -> tuple:
    n = len(bit_array)
    
    # Calculate number of matrices (N)
    bits_per_matrix = M * Q  # 1024 bits per matrix
    N = int(n // bits_per_matrix)
    
    # NIST recommends N >= 38, but we allow N >= 1 for functional API test
    if N < 1:
        # Not enough data for even one matrix
        return 0.0, False
        
    # Discard remaining bits
    T = bit_array[:N * bits_per_matrix]
    
    # V stores observed counts for: [Full Rank (32), Full-1 (31), Full-2 or less (<=30)]
    V = np.zeros(3, dtype=int)
    
    for i in range(N):
        start_idx = i * bits_per_matrix
        end_idx = (i + 1) * bits_per_matrix
        
        matrix_flat = T[start_idx:end_idx]
        matrix = matrix_flat.reshape(M, Q)
        
        R = _binary_matrix_rank(matrix) 
        
        if R == M:      # Rank 32
            V[0] += 1 
        elif R == M - 1: # Rank 31
            V[1] += 1 
        else:           # Rank <= 30
            V[2] += 1 
            
    # Calculate Chi-squared Test Statistic
    # chi^2 = sum( (observed - expected)^2 / expected )
    expected = N * RANK_PROBABILITIES
    
    chi_squared = np.sum((V - expected)**2 / expected)
    
    # Calculate P-value (df=2)
    P_value = chi2.sf(chi_squared, DEGREES_OF_FREEDOM)
    
    verdict = P_value >= alpha
    
    return P_value, verdict