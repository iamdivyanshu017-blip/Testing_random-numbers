import numpy as np
from scipy.stats import chi2

def block_frequency_test(bit_array: np.ndarray, M: int = 10000, alpha: float = 0.01) -> tuple:
    n = len(bit_array)
    
    # Calculate number of full blocks (N)
    N = int(np.floor(n / M))

    # NIST condition: M >= 20, M > 0.01n, N < 100 optional checks.
    # We enforce a simpler functional check: need at least a reasonable number of blocks.
    if N < 20:
        return 0.0, False

    # Discard excess bits
    T = bit_array[:N * M]
    blocks = T.reshape(N, M)

    # Calculate proportion of ones (pi_i) for each block
    # Sum over axis 1 (rows) gives count of ones per block
    pi_i = np.sum(blocks, axis=1) / M

    # Calculate Chi-Square Statistic
    # chi^2 = 4 * M * sum((pi_i - 0.5)^2)
    chi_squared = 4 * M * np.sum((pi_i - 0.5)**2)

    # Calculate P-value
    # Degrees of freedom = N
    P_value = chi2.sf(chi_squared, N) 

    # Determine Verdict
    verdict = P_value >= alpha

    return P_value, verdict