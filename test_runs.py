import numpy as np
import math
from scipy.special import erfc

def runs_test(bit_array: np.ndarray, alpha: float = 0.01) -> tuple:
    n = len(bit_array)
    if n == 0:
        return 0.0, False

    # 1. Compute Proportion of Ones (pi)
    num_ones = np.sum(bit_array)
    pi = num_ones / n
    
    # 2. Prerequisite Check: Frequency (Monobit) Test
    # The Runs test is only valid if the number of ones and zeros is roughly equal.
    # Condition: |pi - 0.5| < 2 / sqrt(n)
    if abs(pi - 0.5) >= (2.0 / math.sqrt(n)):
        return 0.0, False
    
    # 3. Compute Observed Number of Runs (V_n)
    # A run starts if the current bit is different from the next bit.
    # We start with 1 run (the first block of bits).
    v_n = 1
    for i in range(n - 1):
        if bit_array[i] != bit_array[i+1]:
            v_n += 1
    
    # 4. Compute Test Statistic
    # Numerator = |V_n - 2n*pi*(1-pi)|
    numerator = abs(v_n - 2 * n * pi * (1.0 - pi))
    
    # Denominator = 2 * sqrt(2n) * pi * (1-pi)
    denominator = 2.0 * math.sqrt(2.0 * n) * pi * (1.0 - pi)
    
    if denominator == 0:
        return 0.0, False

    # 5. Compute P-value
    P_value = erfc(numerator / denominator)

    # 6. Verdict
    verdict = P_value >= alpha
    
    return P_value, verdict