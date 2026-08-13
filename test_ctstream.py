import numpy as np
from scipy.stats import chi2

def count_the_1s_stream_test(rng_func, num_words=256000):
    """
    Production-level Count-the-1s Stream Test.
    Fixed: Corrected OverflowError by using proper integer casting.
    """
    print(f"Executing Count-the-1s Stream Test: {num_words} words...")

    # 1. Precompute the category for every possible byte (0-255)
    # Categories: <=3:0, 4:1, 5:2, 6:3, >=7:4
    byte_to_cat = np.zeros(256, dtype=np.int32) # Use int32 for the map
    for i in range(256):
        ones = bin(i).count('1')
        if ones <= 3: byte_to_cat[i] = 0
        elif ones == 4: byte_to_cat[i] = 1
        elif ones == 5: byte_to_cat[i] = 2
        elif ones == 6: byte_to_cat[i] = 3
        else: byte_to_cat[i] = 4

    # 2. Generate random bytes
    # To get num_words overlapping words, we need num_words + 4 bytes.
    num_ints = (num_words // 4) + 5
    raw_data = rng_func(num_ints)
    byte_stream = raw_data.view(np.uint8)
    
    # 3. Map bytes to categories and ensure they are int32 to prevent OverflowError
    # This was the specific fix for your traceback.
    categories = byte_to_cat[byte_stream].astype(np.int32)
    
    # 4. Count frequencies of 5-letter words
    # 5^5 = 3125 possible word combinations
    word_counts = np.zeros(3125, dtype=np.int32)
    
    # Sliding window to form 5-letter words
    # We use explicit integer arithmetic to avoid uint8 bounds
    for i in range(num_words):
        idx = (categories[i] * 625 + 
               categories[i+1] * 125 + 
               categories[i+2] * 25 + 
               categories[i+3] * 5 + 
               categories[i+4])
        word_counts[idx] += 1

    # 5. Statistical Analysis
    # p_ones is the binomial probability P(X=k) for n=8, p=0.5
    # Category Probabilities:
    p_cat = np.array([
        (1+8+28+56)/256, # Cat 0: <=3 ones
        70/256,          # Cat 1: 4 ones
        56/256,          # Cat 2: 5 ones
        28/256,          # Cat 3: 6 ones
        (8+1)/256        # Cat 4: >=7 ones
    ])
    
    # Calculate Expected Frequencies
    expected = np.zeros(3125)
    for i in range(3125):
        # Determine the probability of this specific word
        # (e.g., probability of word '01234')
        temp = i
        prob = 1.0
        for _ in range(5):
            prob *= p_cat[temp % 5]
            temp //= 5
        expected[i] = num_words * prob

    # 6. Chi-Square Test
    # Calculate (O-E)^2 / E
    chi_stat = np.sum((word_counts - expected)**2 / expected)
    
    # Standard degrees of freedom for this test in Diehard is 2500,
    # but based on the bins, we use N_bins - 1 = 3124.
    df = 3124
    p_value = 1 - chi2.cdf(chi_stat, df=df)

    print("-" * 40)
    print(f"Total Words Processed: {num_words}")
    print(f"Chi-Square Statistic:  {chi_stat:.4f}")
    print(f"Degrees of Freedom:    {df}")
    print(f"P-Value:               {p_value:.6f}")

    if 0.0001 < p_value < 0.9999:
        print("RESULT: PASSED")
    else:
        print("RESULT: FAILED")
        
    return p_value

# Helper for the RNG
def sys_rng(n):
    return np.random.randint(0, 0xFFFFFFFF, size=n, dtype=np.uint32)

if __name__ == "__main__":
    count_the_1s_stream_test(sys_rng)