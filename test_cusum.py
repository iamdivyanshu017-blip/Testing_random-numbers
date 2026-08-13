import numpy as np
from scipy.stats import norm
from typing import Tuple

def _calculate_cusum_p_value(Z: int, n: int) -> float:
    if Z == 0:
        return 1.0 
        
    sqrt_n = np.sqrt(n)
    
    limit = int(abs(n / Z)) + 10 # Adding buffer
    
    term_1 = 0.0
    for k in range(-limit, limit + 1):
        upper = ((4 * k + 1) * Z) / sqrt_n
        lower = ((4 * k - 1) * Z) / sqrt_n
        term_1 += (norm.cdf(upper) - norm.cdf(lower))
        
    term_2 = 0.0
    for k in range(-limit, limit + 1):
        upper = ((4 * k + 3) * Z) / sqrt_n
        lower = ((4 * k + 1) * Z) / sqrt_n
        term_2 += (norm.cdf(upper) - norm.cdf(lower))

    p_val = 1.0 - term_1 + term_2
    return float(np.clip(p_val, 0.0, 1.0))

def cumulative_sums_test(bit_array: np.ndarray, alpha: float = 0.01) -> Tuple[float, bool, float, bool]:
    n = len(bit_array)
    if n == 0:
        return 0.0, False, 0.0, False
        
    # Convert bits (0,1) to bipolar (-1, 1)
    X = np.where(bit_array == 0, -1, 1) 
    
    # --- Forward Mode ---
    # S_i = X_1 + ... + X_i
    S_k = np.cumsum(X)
    Z_forward = np.max(np.abs(S_k))
    p_forward = _calculate_cusum_p_value(int(Z_forward), n)
    
    # --- Backward Mode ---
    # S_i = X_n + ... + X_{n-i+1}
    S_k_rev = np.cumsum(X[::-1])
    Z_backward = np.max(np.abs(S_k_rev))
    p_backward = _calculate_cusum_p_value(int(Z_backward), n)
    
    return p_forward, p_forward >= alpha, p_backward, p_backward >= alpha