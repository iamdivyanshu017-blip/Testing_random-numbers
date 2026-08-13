import numpy as np
from scipy.stats import chi2

def overlapping_templates_test(bit_array: np.ndarray, alpha: float = 0.01) -> tuple:

    n = len(bit_array)
    
    # NIST Constants for m=9
    M = 1032  # Block size
    m = 9     # Template length (run of ones)
    K = 6     # Number of degrees of freedom + 1 (Bins: 0, 1, 2, 3, 4, >=5)
    
    # Probabilities from NIST SP 800-22 Rev 1a, Table 3.4 (m=9)
    # These correspond to the theoretical Poisson distribution for lambda = 2
    #(updated later:
     # Correct NIST SP 800-22 theoretical probabilities for the Overlapping
     # Templates test (M=1032, m=9), derived from eta, NOT the Poisson(2)
     # table used by the Non-Overlapping Templates test.)
    PI = np.array([0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865])
    
    # Calculate number of blocks
    N = int(n // M)
    
    if N < 5:
        # Not enough data for a statistically valid test (min 5 blocks)
        # Return 0.0 to signal failure to the API
        return 0.0, False

    # Discard excess bits
    used_bits = bit_array[:N * M]
    blocks = used_bits.reshape(N, M)
    
    # Array to store observed frequencies for the bins (0 to 5)
    v_counts = np.zeros(K)
    
    # Template: '111111111'
    template_str = '1' * m
    
    for block in blocks:
        # Optimization: Convert binary block to string for faster searching
        # than iterating with numpy windows.
        block_str = "".join(map(str, block))
        
        count = 0
        start_index = 0
        
        # Count overlapping occurrences
        while True:
            idx = block_str.find(template_str, start_index)
            if idx == -1:
                break
            count += 1
            # For overlapping, we shift by 1. 
            # (Standard Python count() is non-overlapping, so we loop with find)
            start_index = idx + 1
            
        # Increment the appropriate bin
        if count >= 5:
            v_counts[5] += 1
        else:
            v_counts[count] += 1

    # Calculate Chi-squared Statistic
    # chi^2 = Sum [ (Observed - Expected)^2 / Expected ]
    expected_counts = N * PI
    
    # Avoid division by zero warnings if N is very small (though checked above)
    chi_squared = np.sum((v_counts - expected_counts)**2 / expected_counts)
    
    # Calculate P-value (df = K - 1 = 5)
    P_value = chi2.sf(chi_squared, K - 1)
    
    verdict = P_value >= alpha
    
    return P_value, verdict
