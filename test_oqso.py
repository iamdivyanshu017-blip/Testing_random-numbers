import numpy as np
from scipy.stats import norm

def oqso_test(rng_func, num_words=2**21):
    """
    Production-level OQSO (Overlapping-Quadruplets-Sparse-Occupancy) Test.
    
    Args:
        rng_func: Function returning random 32-bit integers.
        num_words: Number of overlapping quadruplets to test (Standard is 2^21).
    """
    print(f"Executing OQSO Test: Testing {num_words} overlapping quadruplets...")

    # 1. Initialize State Space
    # 2^20 possible quadruplets (32^4)
    num_cells = 1 << 20
    occupancy = np.zeros(num_cells, dtype=bool)
    
    # To get num_words overlapping quadruplets, we need num_words + 3 integers
    # Quad 1: [Int 0, 1, 2, 3], Quad 2: [Int 1, 2, 3, 4], etc.
    raw_data = rng_func(num_words + 3)
    
    def get_5bit_letter(val):
        # Extract a 5-bit letter. Diehard usually takes bits 20-24.
        return (val >> 20) & 0x1F

    # 2. Extract all letters first for speed
    letters = np.array([get_5bit_letter(x) for x in raw_data], dtype=np.uint32)

    # 3. Sliding Window for Quadruplets
    # We combine 4 consecutive 5-bit letters into a single 20-bit index
    for i in range(num_words):
        # Index = (L1 << 15) | (L2 << 10) | (L3 << 5) | L4
        # We can optimize this using bitwise shifts on the letter array
        cell_idx = (letters[i] << 15) | (letters[i+1] << 10) | (letters[i+2] << 5) | letters[i+3]
        occupancy[cell_idx] = True

    # 4. Count Missing Quadruplets
    missing_cells = num_cells - np.count_nonzero(occupancy)

    # 5. Statistical Constants for OQSO
    # Marsaglia's constants for 2^21 trials and 2^20 cells:
    # Expected mean: 141,909
    # Expected standard deviation: 295 (Note: slightly different from OPSO)
    expected_mean = 141909
    expected_stddev = 295
    
    z_score = (missing_cells - expected_mean) / expected_stddev
    p_value = 1 - norm.cdf(z_score)

    print("-" * 30)
    print(f"Total Possible Quads: {num_cells}")
    print(f"Observed Missing:      {missing_cells}")
    print(f"Expected Missing:      {expected_mean}")
    print(f"Z-Score:               {z_score:.4f}")
    print(f"P-Value:               {p_value:.6f}")

    if 0.0001 < p_value < 0.9999:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
        
    return p_value

# Example Usage
def mock_rng(n):
    return np.random.randint(0, 0xFFFFFFFF, size=n, dtype=np.uint32)

oqso_test(mock_rng)