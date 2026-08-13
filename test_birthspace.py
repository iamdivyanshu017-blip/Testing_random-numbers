import numpy as np
from scipy.stats import poisson, chi2

def birthday_spacings_test(rng_func, m=512, n=2**24, trials=1000):

    # Theoretical Mean: Derived from: lambda = m^3 / (4 * n)
    lam = (m**3) / (4 * n)
    
    duplicate_counts = []

    for _ in range(trials):
        birthdays = rng_func(m) % n
        birthdays.sort()
        
        spacings = np.diff(birthdays)
        spacings.sort()
        
        duplicates = np.sum(np.diff(spacings) == 0)
        duplicate_counts.append(duplicates)

    total_duplicates = sum(duplicate_counts)
    total_lambda = trials * lam
    
    # A very small p-value means too many duplicates (non-random clustering)
    # A very high p-value means too few duplicates (non-random uniformity)
    p_value = 1 - poisson.cdf(total_duplicates, total_lambda)
    
    # Report Results
    print(f"--- Diehard Birthday Spacings Result ---")
    print(f"Trials: {trials} | Birthdays: {m} | Range: {n}")
    print(f"Expected Duplicates: {total_lambda:.2f}")
    print(f"Observed Duplicates: {total_duplicates}")
    print(f"P-Value: {p_value:.6f}")
    
    if 0.0001 < p_value < 0.9999:
        print("RESULT: PASSED")
    else:
        print("RESULT: FAILED (Non-Random)")
        
    return p_value

# Example Usage with a standard generator
def mock_rng(size):
    return np.random.randint(0, 2**32, size=size, dtype=np.uint32)

birthday_spacings_test(mock_rng)