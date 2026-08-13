import numpy as np
from scipy.stats import chi2

def count_the_1s_specific_test(rng_func, num_words=256000):
    """
    Production-level Count-the-1s Specific Test.
    Targets specific bytes within the 32-bit word.
    """
    print(f"Executing Count-the-1s Specific Test: {num_words} words...")

    # 1. Precompute category map (Popcount -> Category)
    byte_to_cat = np.zeros(256, dtype=np.int32)
    for i in range(256):
        ones = bin(i).count('1')
        if ones <= 3: byte_to_cat[i] = 0
        elif ones == 4: byte_to_cat[i] = 1
        elif ones == 5: byte_to_cat[i] = 2
        elif ones == 6: byte_to_cat[i] = 3
        else: byte_to_cat[i] = 4

    # 2. Generate random integers
    # Each integer provides 2 letters. To get N words of 5 letters each,
    # we need (N + 4) letters total.
    num_ints_needed = (num_words + 5) // 2 + 1
    raw_data = rng_func(num_ints_needed)

    # 3. Extract specific bytes (e.g., bits 16-23 and 24-31)
    # This targets the 'upper' half of the 32-bit word.
    byte_a = (raw_data >> 16) & 0xFF
    byte_b = (raw_data >> 24) & 0xFF
    
    # Interleave them to form the letter stream: [A1, B1, A2, B2, ...]
    letter_stream = np.empty(len(raw_data) * 2, dtype=np.int32)
    letter_stream[0::2] = byte_to_cat[byte_a]
    letter_stream[1::2] = byte_to_cat[byte_b]

    # 4. Count frequencies of 5-letter words
    word_counts = np.zeros(3125, dtype=np.int32)
    
    # Sliding window
    for i in range(num_words):
        idx = (letter_stream[i] * 625 + 
               letter_stream[i+1] * 125 + 
               letter_stream[i+2] * 25 + 
               letter_stream[i+3] * 5 + 
               letter_stream[i+4])
        word_counts[idx] += 1

    # 5. Statistical Analysis
    p_cat = np.array([(1+8+28+56)/256, 70/256, 56/256, 28/256, (8+1)/256])
    
    expected = np.zeros(3125)
    for i in range(3125):
        temp = i
        prob = 1.0
        for _ in range(5):
            prob *= p_cat[temp % 5]
            temp //= 5
        expected[i] = num_words * prob

    # 6. Chi-Square Test
    chi_stat = np.sum((word_counts - expected)**2 / expected)
    df = 3124
    p_value = 1 - chi2.cdf(chi_stat, df=df)

    print("-" * 40)
    print(f"Targeting:      Upper 16 bits (two 8-bit segments)")
    print(f"Chi-Square Stat: {chi_stat:.4f}")
    print(f"P-Value:         {p_value:.6f}")

    if 0.0001 < p_value < 0.9999:
        print("RESULT: PASSED")
    else:
        print("RESULT: FAILED")
        
    return p_value

def sample_rng(n):
    return np.random.randint(0, 0xFFFFFFFF, size=n, dtype=np.uint32)

if __name__ == "__main__":
    count_the_1s_specific_test(sample_rng)