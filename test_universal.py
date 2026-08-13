import numpy as np
import math
from scipy.stats import norm

# --- NIST CONSTANTS (Corrected for L=7) ---
# Source: NIST SP 800-22 Rev 1a, Table 3.3
L = 7                  
Q = 10 * 2**L  # Q = 1280
MU = 6.1962507 # Expected Value for L=7
VARIANCE = 3.125 # Variance for L=7

def _get_int_from_block(block: np.ndarray) -> int:
    """Converts a binary block (array of 0s/1s) to an integer."""
    val = 0
    for bit in block:
        val = (val << 1) | int(bit)
    return val

def universal_test(bit_array: np.ndarray, alpha: float = 0.01) -> tuple:
    n = len(bit_array)
    
    # Calculate number of test blocks (K)
    # Total blocks = floor(n / L)
    # K = Total blocks - Q (initialization blocks)
    K = int(np.floor(n / L)) - Q
    
    if K <= 0:
        # Not enough data to run the test
        # For L=7, minimum n approx (1280 + 1) * 7 = 8967 bits
        return 0.0, False, K
        
    # Create the table of L-bit blocks
    # We use only the bits that fit into full blocks
    num_blocks = Q + K
    T_bits = bit_array[:num_blocks * L]
    blocks = T_bits.reshape(num_blocks, L)
    
    # Table to store the LAST time a pattern was seen
    # Size is 2^L (128 for L=7). Initialize with 0.
    last_seen_table = np.zeros(2**L, dtype=int) 
    
    # 1. Initialization Phase (First Q blocks)
    for i in range(Q):
        pattern_int = _get_int_from_block(blocks[i])
        last_seen_table[pattern_int] = i + 1  # Store 1-based index (block number)
        
    # 2. Test Phase (Next K blocks)
    sum_log = 0.0
    for i in range(Q, num_blocks):
        pattern_int = _get_int_from_block(blocks[i])
        
        # Calculate distance from the last time this pattern was seen
        # Note: pattern_int is guaranteed to be in last_seen_table because Q is large enough
        # by design, but we use the table value regardless.
        last_index = last_seen_table[pattern_int]
        
        # Distance = Current Position - Last Position
        distance = (i + 1) - last_index
        
        # Accumulate log2 of the distance
        sum_log += math.log2(distance)
        
        # Update the table with the current position
        last_seen_table[pattern_int] = i + 1
      
    # 3. Compute Statistics
    phi = sum_log / K
    
    # Standard deviation of the mean = sqrt(Variance / K)
    sigma_of_mean = math.sqrt(VARIANCE / K)
    
    # Z-Score
    # Z = (phi - expected_value) / sigma_of_mean
    if sigma_of_mean == 0:
        return 0.0, False, K
        
    Z = (phi - MU) / sigma_of_mean
    
    # P-value calculation (Two-sided)
    # P = 2 * (1 - CDF(|Z|)) = 2 * sf(|Z|)
    P_value = norm.sf(np.abs(Z)) * 2.0
    
    verdict = P_value >= alpha
    
    return P_value, verdict, K