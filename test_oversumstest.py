import numpy as np
from scipy.stats import norm, kstest

def overlapping_sums_test(rng_func, n=1000000, window_size=10):
    """
    Production-level Overlapping Sums Test.
    
    Args:
        rng_func: Function returning random floats in [0, 1).
        n: Number of random floats to generate (default 1 million).
        window_size: Number of floats per sum (Diehard standard is 10).
    """
    print(f"Executing Overlapping Sums Test: n={n}, window={window_size}...")

    # 1. Generate the sequence of random floats
    u = rng_func(n)

    # 2. Calculate Overlapping Sums
    # We use a convolution (rolling sum) for O(N) performance
    # 'valid' ensures we only get sums of exactly window_size
    sums = np.convolve(u, np.ones(window_size), mode='valid')

    # 3. Theoretical parameters for the sum of 'window_size' U(0,1) variables
    # Mean of one U(0,1) is 0.5, Variance is 1/12
    theoretical_mean = window_size * 0.5
    theoretical_std = np.sqrt(window_size / 12.0)

    # 4. Transform to Standard Normal (Z-scores)
    # Z = (X - mu) / sigma
    z_scores = (sums - theoretical_mean) / theoretical_std

    # 5. Statistical Analysis
    # Since they are overlapping, we don't just use a simple Chi-Square on all.
    # However, for a sufficiently large N, we can test the distribution 
    # of the resulting Z-scores against a Standard Normal N(0,1).
    # We use the Kolmogorov-Smirnov test.
    
    ks_stat, p_value = kstest(z_scores, 'norm')

    print("-" * 40)
    print(f"Number of Sums:   {len(sums)}")
    print(f"Observed Mean:    {np.mean(sums):.6f}")
    print(f"Expected Mean:    {theoretical_mean:.1f}")
    print(f"KS Statistic:     {ks_stat:.4f}")
    print(f"P-Value:          {p_value:.6f}")

    if 0.0001 < p_value < 0.9999:
        print("RESULT: PASSED")
    else:
        print("RESULT: FAILED")
        
    return p_value

# Helper for the RNG
def standard_rng(n):
    return np.random.random(n)

if __name__ == "__main__":
    overlapping_sums_test(standard_rng)