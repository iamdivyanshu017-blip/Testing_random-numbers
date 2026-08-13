import numpy as np
import math
from scipy.special import erfc

def bipolar_sequence(bit_array: np.ndarray) -> np.ndarray:
    
    return 2 * bit_array - 1

def frequency_monobit_test(bit_array: np.ndarray, alpha: float = 0.01) -> tuple:
    
    n = len(bit_array)
    if n == 0:
        return 0.0, False

    # 1. Convert to bipolar sequence (X_i = -1 or +1)
    X = bipolar_sequence(bit_array)
    
    # 2. Calculate the test statistic (S_n = sum(X_i))
    S_n = np.sum(X)
    
    # 3. Calculate P-value
    # P = erfc( |S_n| / sqrt(2*n) )
    # erfc is the complementary error function.
    denominator = math.sqrt(2 * n)
    P_value = erfc(abs(S_n) / denominator)
    
    # 4. Determine verdict
    verdict = P_value >= alpha
    
    return P_value, verdict
