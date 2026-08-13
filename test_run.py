import numpy as np
from scipy.stats import chi2

def run_counts(data):
    """Counts the lengths of runs up in a sequence."""
    n = len(data)
    counts = np.zeros(6) # Bins for lengths 1, 2, 3, 4, 5, >=6
    
    i = 0
    while i < n - 1:
        run_len = 1
        while i < n - 1 and data[i+1] > data[i]:
            run_len += 1
            i += 1
        
        # Increment the appropriate bin
        bin_idx = min(run_len, 6) - 1
        counts[bin_idx] += 1
        i += 1 # Move to start of next run
    return counts

def runs_test(rng_func, n=100000):
    """
    Production-level Runs Test (Up and Down).
    """
    print(f"Executing Runs Test: n={n}...")
    
    data = rng_func(n)
    
    # 1. Get counts for Runs Up
    up_counts = run_counts(data)
    
    # 2. Get counts for Runs Down (by flipping the sequence)
    down_counts = run_counts(-data)
    
    # 3. Marsaglia's Inverse Covariance Matrix (A) for Runs Test
    # This matrix accounts for the dependency between adjacent runs
    A = np.array([
        [4529.4, 9044.9, 13568.0, 18091.0, 22615.0, 27892.0],
        [9044.9, 18097.0, 27139.0, 36187.0, 45234.0, 55789.0],
        [13568.0, 27139.0, 40721.0, 54281.0, 67852.0, 83685.0],
        [18091.0, 36187.0, 54281.0, 72414.0, 90470.0, 111580.0],
        [22615.0, 45234.0, 67852.0, 90470.0, 113262.0, 139476.0],
        [27892.0, 55789.0, 83685.0, 111580.0, 139476.0, 172860.0]
    ])
    
    # Theoretical Expected counts for runs of length 1..6
    # Based on: E_i = n * Prob(run_len = i)
    expected_probs = np.array([1/6, 5/24, 11/120, 19/720, 29/5040, 1/840])
    expected = n * expected_probs
    
    def calculate_p(counts):
        diffs = counts - expected
        # Quadratic form: diffs^T * A * diffs / n
        # Note: A is actually 1/n * InverseCovariance
        v = np.dot(np.dot(diffs, A), diffs) / n
        return 1 - chi2.cdf(v, df=6)

    p_up = calculate_p(up_counts)
    p_down = calculate_p(down_counts)

    print("-" * 40)
    print(f"Runs Up P-Value:   {p_up:.6f}")
    print(f"Runs Down P-Value: {p_down:.6f}")

    if 0.0001 < p_up < 0.9999 and 0.0001 < p_down < 0.9999:
        print("RESULT: PASSED")
    else:
        print("RESULT: FAILED")
        
    return p_up, p_down

# RNG helper
def standard_rng(n):
    return np.random.random(n)

if __name__ == "__main__":
    runs_test(standard_rng)