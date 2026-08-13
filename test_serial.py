import numpy as np
from scipy.stats import chi2
from typing import Dict

def serial_test(bit_array: np.ndarray, m: int = 16, alpha: float = 0.01) -> tuple:
    n = len(bit_array)
    
    # NIST Requirement: n should be significantly larger than 2^m
    # Ideally n >= m * 2^m, but soft check is n >= 2^(m+1)
    if n < 2**(m + 1):
        return 0.0, False, 0.0, False

    def get_psi_sq(block_len: int) -> float:
        if block_len == 0: return 0.0
        
        # 1. Augment the sequence for circularity
        # Append the first block_len - 1 bits to the end
        if block_len > 1:
            aug_bits = np.concatenate((bit_array, bit_array[:block_len-1]))
        else:
            aug_bits = bit_array
            
        # 2. Count Occurrences
        # Using integer representation for efficiency with larger m
        # (Though m=16 is large, standard python dict handles 2^16 keys fine)
        counts = {}
        mask = (1 << block_len) - 1
        val = 0
        
        # Pre-fill first window
        for i in range(block_len):
            val = (val << 1) | aug_bits[i]
            
        counts[val] = 1
        
        for i in range(1, n):
            # Shift left, drop high bit, add new low bit
            val = ((val << 1) & mask) | aug_bits[i + block_len - 1]
            counts[val] = counts.get(val, 0) + 1
            
        # 3. Calculate Sum of Squares
        sum_sq = sum(c**2 for c in counts.values())
        
        # 4. Calculate Psi^2
        # psi^2 = (2^m / n) * sum(counts^2) - n
        psi_sq_val = (pow(2, block_len) / n) * sum_sq - n
        
        return psi_sq_val

    # Calculate Psi^2 statistics for m, m-1, and m-2
    psi_sq_m   = get_psi_sq(m)
    psi_sq_m_1 = get_psi_sq(m-1)
    psi_sq_m_2 = get_psi_sq(m-2)
    
    # --- Statistic 1 (Delta 1) ---
    delta1 = psi_sq_m - psi_sq_m_1
    df1 = 2**(m-1)
    P_value_1 = chi2.sf(delta1, df1)
    
    # --- Statistic 2 (Delta 2) ---
    delta2 = psi_sq_m - 2*psi_sq_m_1 + psi_sq_m_2
    df2 = 2**(m-2)
    P_value_2 = chi2.sf(delta2, df2)
    
    verdict_1 = P_value_1 >= alpha
    verdict_2 = P_value_2 >= alpha
    
    return P_value_1, verdict_1, P_value_2, verdict_2