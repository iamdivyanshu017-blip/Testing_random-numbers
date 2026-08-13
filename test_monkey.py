import numpy as np
from scipy.stats import norm

def bitstream_monkey_test(rng_func, word_len=20, num_bits=2**21):
    """
    Production-level Bitstream Monkey Test (OPSO/OQSO/DNA style).
    
    Args:
        rng_func: Function returning random 32-bit integers.
        word_len: Length of the 'word' in bits (default 20).
        num_bits: Total bits to examine (default 2^21).
    """
    print(f"Starting Monkey Test: {word_len}-bit words, {num_bits} bits...")

    # 1. Prepare bitstream
    # We need enough 32-bit ints to cover num_bits
    n_ints = (num_bits // 32) + 1
    raw_data = rng_func(n_ints)
    
    # 2. Use a bitset to track 'seen' words
    # There are 2^word_len possible words (for 20 bits, this is 1,048,576)
    num_possible_words = 1 << word_len
    seen_words = np.zeros(num_possible_words, dtype=bool)

    # 3. Sliding Window over bits
    # We extract overlapping word_len bits
    # To optimize, we use a rolling mask
    current_word = 0
    bit_count = 0
    words_found = 0
    
    mask = num_possible_words - 1
    
    for val in raw_data:
        for shift in range(32):
            if bit_count >= num_bits:
                break
            
            # Slide one bit into the word
            bit = (val >> shift) & 1
            current_word = ((current_word << 1) | bit) & mask
            
            # We need to wait until we have a full word_len before counting
            if bit_count >= word_len:
                if not seen_words[current_word]:
                    seen_words[current_word] = True
                    words_found += 1
            
            bit_count += 1

    # 4. Statistical Analysis
    # The number of MISSING words (not found)
    missing_words = num_possible_words - words_found
    
    # For word_len=20 and 2^21 bits, the theoretical mean and variance:
    # Expected missing (Exp[-lambda] * 2^L) where lambda = bits/2^L
    # For Diehard parameters: Mean approx 141909, StdDev approx 290
    lambd = (num_bits - word_len + 1) / num_possible_words
    expected_mean = num_possible_words * np.exp(-lambd)
    
    # Variance for overlapping bitstream is complex; 
    # Dieharder uses pre-calculated constants for the standard 20-bit test.
    expected_stddev = 290 # Standard Diehard constant for these parameters
    
    z_score = (missing_words - expected_mean) / expected_stddev
    p_value = 1 - norm.cdf(z_score)

    print("-" * 30)
    print(f"Possible Words: {num_possible_words}")
    print(f"Missing Words:  {missing_words}")
    print(f"Expected Mean:  {expected_mean:.2f}")
    print(f"Z-Score:        {z_score:.4f}")
    print(f"P-Value:        {p_value:.6f}")

    if 0.0001 < p_value < 0.9999:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")

    return p_value

# Example Usage:
def secure_rng(n):
    return np.random.randint(0, 0xFFFFFFFF, size=n, dtype=np.uint32)

bitstream_monkey_test(secure_rng)