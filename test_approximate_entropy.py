import numpy as np
import math
from scipy.stats import chi2

def approximate_entropy_test(bit_array: np.ndarray, m: int = 10, alpha: float = 0.01) -> tuple:
    n = len(bit_array)
    
    # Ideally n >= 2^m * 10, but strict lower bound check is often n >= 2^(m+1).
    if n < (1 << (m + 1)):
        return 0.0, False

    def _get_phi(m_len: int) -> float:
        """
        Calculates Phi(m) = sum( (C_i / n) * ln(C_i / n) )
        """
        # 1. Augment the sequence (circularity)
        # Append first m_len-1 bits to the end
        extended_bits = np.concatenate((bit_array, bit_array[:m_len-1]))
        
        # 2. Vectorized Pattern Extraction
        # We need to convert every overlapping block of length m_len into an integer.
        # Instead of a slow loop, we use numpy shifting.
        
        # Initialize an array of zeros with the same length as the original sequence
        powers = np.zeros(n, dtype=int)
        
        for i in range(m_len):
            # Shift accumulated values left
            powers <<= 1
            # Add the bit at the current offset (slice from i to i+n)
            # This effectively builds the integer value for every window position simultaneously
            powers |= extended_bits[i : i + n]
            
        # 3. Count Frequencies
        # np.unique is highly optimized for this
        _, counts = np.unique(powers, return_counts=True)
        
        # 4. Compute Phi
        # pi = C_i / n
        # phi = sum(pi * ln(pi))
        C_i = counts.astype(float)
        pi = C_i / n
        
        # Using natural log (np.log) as per NIST spec
        phi_sum = np.sum(pi * np.log(pi))
        
        return phi_sum

    # Calculate phi(m) and phi(m+1)
    phi_m = _get_phi(m)
    phi_m_plus_1 = _get_phi(m + 1)
    
    # ApEn(m) = phi(m) - phi(m+1)
    apen = phi_m - phi_m_plus_1
    
    # Chi-Squared Statistic
    # chi^2 = 2n * (ln(2) - ApEn(m))
    chi_squared = 2 * n * (math.log(2) - apen)
    
    # Degrees of Freedom = 2^m
    df = 1 << m
    
    # P-value = chi2.sf(chi_squared, df)
    p_value = chi2.sf(chi_squared, df)
    
    verdict = p_value >= alpha
    
    return p_value, verdict