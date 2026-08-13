import numpy as np
from scipy.stats import norm

def dna_test(rng_func, num_words=2**21):
    """
    Production-level DNA Test (Monkey Test on 10-letter words).
    
    Args:
        rng_func: Function returning random 32-bit integers.
        num_words: Total overlapping 10-letter words to test (Standard is 2^21).
    """
    print(f"Executing DNA Test: Testing {num_words} overlapping 10-letter words...")

    # 1. State Space Setup
    # 4^10 = 2^20 possible DNA sequences
    num_cells = 1 << 20
    occupancy = np.zeros(num_cells, dtype=bool)
    
    # We need enough bits to form num_words + (word_len - 1) letters
    # Each letter is 2 bits. Total bits needed approx 2 * 2^21.
    # 2^22 bits / 32 bits per int = 2^17 integers.
    num_ints = (num_words >> 4) + 2 
    raw_data = rng_func(num_ints)
    
    # 2. Extract 2-bit letters and create overlapping words
    # A word is 10 letters = 20 bits.
    # We use a 20-bit mask to slide through the bitstream.
    mask = num_cells - 1
    current_window = 0
    words_processed = 0
    
    for val in raw_data:
        # We process each 32-bit integer in 2-bit increments (16 letters per int)
        for i in range(16):
            # Extract 2 bits
            letter = (val >> (i * 2)) & 0x03
            
            # Slide letter into the 20-bit window
            current_window = ((current_window << 2) | letter) & mask
            
            # Start counting once the window is full (10 letters / 20 bits)
            if words_processed >= 10:
                occupancy[current_window] = True
            
            words_processed += 1
            if words_processed >= num_words + 10:
                break
        if words_processed >= num_words + 10:
            break

    # 3. Statistical Analysis
    missing_cells = num_cells - np.count_nonzero(occupancy)

    # 4. Statistical Constants for DNA Test
    # For 2^21 trials and 2^20 cells:
    # Expected mean: 141,909
    # Expected standard deviation: 339 (Specific to 2-bit alphabet overlap)
    expected_mean = 141909
    expected_stddev = 339
    
    z_score = (missing_cells - expected_mean) / expected_stddev
    p_value = 1 - norm.cdf(z_score)

    print("-" * 30)
    print(f"Total Possible DNA Words: {num_cells}")
    print(f"Observed Missing:         {missing_cells}")
    print(f"Expected Missing:         {expected_mean}")
    print(f"Z-Score:                  {z_score:.4f}")
    print(f"P-Value:                  {p_value:.6f}")

    if 0.0001 < p_value < 0.9999:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
        
    return p_value

# Example Usage
def sample_rng(n):
    return np.random.randint(0, 0xFFFFFFFF, size=n, dtype=np.uint32)

dna_test(sample_rng)