import numpy as np
from scipy.stats import chi2

def squeeze_test(rng_func, num_trials=100000):
    """
    Production-level Squeeze Test.
    
    Args:
        rng_func: Function returning random floats in [0, 1).
        num_trials: Number of independent squeeze sequences (Diehard standard: 100k).
    """
    print(f"Executing Squeeze Test: {num_trials} trials...")

    # Theoretical probabilities for j (number of squeezes to reach 1)
    # These constants are derived from the density of products of U(0,1)
    # Bins are for j = 6, 7, 8, 9, ..., 47+
    # Note: Probabilities for j < 6 and j > 47 are near zero.
    
    # Dieharder/Marsaglia standard table for K=2^31
    p = np.array([
        .000000, .000000, .000000, .000000, .000000, .000000,
        .000006, .000061, .000403, .001804, .005939, .015154,
        .031533, .054454, .079872, .101861, .114920, .116248,
        .106670, .089307, .068803, .048998, .032515, .020164,
        .011680, .006323, .003204, .001518, .000672, .000280,
        .000109, .000040, .000014, .000005, .000002, .000001
    ])
    
    # We'll use a count array for j from 6 to 41. 
    # Values outside this range are extremely rare for K=2^31.
    counts = np.zeros(len(p))
    
    # Buffer random numbers to avoid function call overhead
    # Average j is ~18-19, so we need roughly 2 million floats
    raw_floats = rng_func(num_trials * 25)
    float_idx = 0

    for _ in range(num_trials):
        k = 2147483648 # 2^31
        j = 0
        while k > 1:
            if float_idx >= len(raw_floats):
                # Refresh buffer if empty
                raw_floats = rng_func(num_trials * 10)
                float_idx = 0
            
            k = np.ceil(k * raw_floats[float_idx])
            float_idx += 1
            j += 1
        
        # Bin the result (Indices 6 to 41)
        if 6 <= j < 6 + len(p):
            counts[j-6] += 1

    # Chi-Square Test
    expected = p * num_trials
    
    # Filter bins where expected frequency is too low for Chi-Square (> 5)
    valid_mask = expected > 5
    obs = counts[valid_mask]
    exp = expected[valid_mask]
    
    chi_stat = np.sum((obs - exp)**2 / exp)
    df = len(obs) - 1
    p_value = 1 - chi2.cdf(chi_stat, df=df)

    print("-" * 40)
    print(f"Total Trials:     {num_trials}")
    print(f"Chi-Square Stat:  {chi_stat:.4f}")
    print(f"Degrees of Freedom: {df}")
    print(f"P-Value:          {p_value:.6f}")

    if 0.0001 < p_value < 0.9999:
        print("RESULT: PASSED")
    else:
        print("RESULT: FAILED")
        
    return p_value

# Helper for RNG
def standard_rng(n):
    return np.random.random(n)

if __name__ == "__main__":
    squeeze_test(standard_rng)