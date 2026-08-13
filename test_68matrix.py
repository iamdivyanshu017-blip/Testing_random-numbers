import numpy as np
from scipy.stats import chi2

def get_binary_rank_6x8(rows):
    """
    Computes the rank of a 6x8 binary matrix over GF(2).
    'rows' should be a list/array of 6 integers (each treated as 8-bit).
    """
    rank = 0
    # Use 8-bit mask since we are testing 6x8
    matrix = np.array(rows, dtype=np.uint8)
    
    for i in range(8): # Columns 0 to 7
        if rank >= 6: break # Max rank for 6x8 is 6
        
        pivot_mask = 1 << (7 - i)
        
        # Find pivot row
        pivot_row = -1
        for r in range(rank, 6):
            if matrix[r] & pivot_mask:
                pivot_row = r
                break
        
        if pivot_row != -1:
            # Swap
            matrix[rank], matrix[pivot_row] = matrix[pivot_row], matrix[rank]
            # Eliminate
            for r in range(6):
                if r != rank and (matrix[r] & pivot_mask):
                    matrix[r] ^= matrix[rank]
            rank += 1
            
    return rank

def matrix_rank_6x8_test(rng_func, num_matrices=100000):
    """
    Production-level 6x8 Matrix Rank Test.
    
    Args:
        rng_func: Returns random 32-bit integers.
        num_matrices: Number of matrices to test (Diehard uses 100k).
    """
    # Probabilities for Rank 6, 5, and <=4
    p = np.array([0.773115, 0.217439, 0.009446])
    counts = np.zeros(3)

    print(f"Executing 6x8 Matrix Rank Test: {num_matrices} trials...")

    # For efficiency, we grab many integers at once
    # Each 32-bit int can provide four 8-bit rows
    raw_data = rng_func(int(num_matrices * 1.5)) 
    byte_stream = raw_data.view(np.uint8)
    
    idx = 0
    for _ in range(num_matrices):
        # Extract 6 bytes to form the 6x8 matrix
        if idx + 6 > len(byte_stream): break
        matrix_rows = byte_stream[idx : idx + 6]
        idx += 6
        
        r = get_binary_rank_6x8(matrix_rows)
        
        if r == 6:
            counts[0] += 1
        elif r == 5:
            counts[1] += 1
        else:
            counts[2] += 1

    # Chi-Squared Test
    expected = p * num_matrices
    chi_stat = np.sum((counts - expected)**2 / expected)
    p_value = 1 - chi2.cdf(chi_stat, df=2)

    print("-" * 30)
    print(f"Rank 6  (Exp: {expected[0]:.1f}): {int(counts[0])}")
    print(f"Rank 5  (Exp: {expected[1]:.1f}): {int(counts[1])}")
    print(f"Rank <=4 (Exp: {expected[2]:.1f}): {int(counts[2])}")
    print(f"P-Value: {p_value:.6f}")

    if 0.0001 < p_value < 0.9999:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
    
    return p_value

# Example: Testing a standard generator
def system_rng(n):
    return np.random.randint(0, 0xFFFFFFFF, size=n, dtype=np.uint32)

matrix_rank_6x8_test(system_rng)