import numpy as np
from scipy.stats import chi2

def longest_run_test(bit_array: np.ndarray, alpha: float = 0.01) -> tuple:
    n = len(bit_array)
    
    # NIST Standard Parameter for this specific probability table
    M = 128 
    
    # NIST Probabilities for M=128
    # Classes: <=4, 5, 6, 7, 8, >=9
    PI = np.array([0.1174036, 0.2429560, 0.2493635, 0.1751771, 0.1027011, 0.1123988])
    K = 6 # Number of classes
    
    # Calculate number of blocks
    N = int(n // M)
    
    if N < 1:
        # Not enough data for even one block
        return 0.0, False
        
    # Discard excess bits
    used_bits = bit_array[:N * M]
    blocks = used_bits.reshape(N, M)
    
    # Array to store counts for the 6 categories
    v_counts = np.zeros(K)
    
    for block in blocks:
        # Find longest run of ones in this block
        max_run = 0
        current_run = 0
        for bit in block:
            if bit == 1:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 0
                
        # Map run length to category index
        if max_run <= 4:
            idx = 0
        elif max_run == 5:
            idx = 1
        elif max_run == 6:
            idx = 2
        elif max_run == 7:
            idx = 3
        elif max_run == 8:
            idx = 4
        else: # >= 9
            idx = 5
            
        v_counts[idx] += 1
        
    # Calculate Chi-Squared Statistic
    # chi^2 = Sum [ (Observed - Expected)^2 / Expected ]
    expected_counts = N * PI
    
    # Check for potential division by zero if N is very small (though checked above)
    chi_squared = np.sum((v_counts - expected_counts)**2 / expected_counts)
    
    # Calculate P-value (df = K - 1 = 5)
    P_value = chi2.sf(chi_squared, K - 1)
    
    verdict = P_value >= alpha
    
    return P_value, verdict