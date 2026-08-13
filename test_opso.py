import numpy as np
from scipy.stats import norm

def opso_test(rng_func, num_words=2**21):
    """
    Production-level OPSO (Overlapping-Pairs-Sparse-Occupancy) Test.
    
    Args:
        rng_func: Function returning random 32-bit integers.
        num_words: Number of overlapping pairs to test (Standard is 2^21).
    """
    print(f"Executing OPSO Test: Testing {num_words} overlapping pairs...")

    # 1. Initialization
    # 2^20 possible pairs (1024 * 1024)
    num_cells = 1 << 20
    occupancy = np.zeros(num_cells, dtype=bool)
    
    # We need num_words + 1 integers to create num_words overlapping pairs
    # (Pair 1: Int 0 & 1, Pair 2: Int 1 & 2, etc.)
    raw_data = rng_func(num_words + 1)
    
    # 2. Extract Letters and Map to Cells
    # Letter 1: bits 23-14 | Letter 2: bits 13-4 (Standard Diehard extraction)
    # We create overlapping pairs from consecutive 32-bit integers
    
    def get_letter(val):
        # Extract a 10-bit letter from the middle/top of the 32-bit word
        return (val >> 10) & 0x3FF

    # 3. Fill the 'Sparse' Grid
    # An overlapping pair is formed by (Letter_from_Int_i, Letter_from_Int_i+1)
    current_letter = get_letter(raw_data[0])
    
    for i in range(1, len(raw_data)):
        next_letter = get_letter(raw_data[i])
        
        # Combine two 10-bit letters into one 20-bit cell index
        cell_idx = (current_letter << 10) | next_letter
        occupancy[cell_idx] = True
        
        current_letter = next_letter

    # 4. Count Empty Cells
    missing_cells = num_cells - np.count_nonzero(occupancy)

    # 5. Statistical Constants for OPSO (Marsaglia's Constants)
    # For 2^21 pairs and 2^20 cells:
    # Expected mean missing cells: 141,909
    # Expected standard deviation: 290
    expected_mean = 141909
    expected_stddev = 290
    
    z_score = (missing_cells - expected_mean) / expected_stddev
    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    print("-" * 30)
    print(f"Total Possible Pairs: {num_cells}")
    print(f"Observed Missing:     {missing_cells}")
    print(f"Expected Missing:     {expected_mean}")
    print(f"Z-Score:              {z_score:.4f}")
    print(f"P-Value:              {p_value:.6f}")

    if 0.0001 < p_value < 0.9999:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
        
    return p_value

# Example Usage
def pcg64_rng(n):
    return np.random.Generator(np.random.PCG64()).integers(0, 2**32, size=n, dtype=np.uint32)

opso_test(pcg64_rng)