import numpy as np
import math
from scipy.fft import fft
from scipy.special import erfc

def spectral_test(bit_array: np.ndarray, alpha: float = 0.01) -> tuple:
    n = len(bit_array)
    
    # 1. Minimum length check
    # NIST recommends n >= 1000 for this test to be valid.
    if n < 1000:
        return 0.0, False

    # 2. Convert to Bipolar sequence {-1, 1}
    # 0 -> -1, 1 -> +1
    X = 2 * bit_array - 1

    # 3. Compute Discrete Fourier Transform (DFT)
    # We only need the first n/2 components (M) because the spectrum is symmetric
    S = fft(X)
    M = n // 2
    
    # Calculate Moduli (Magnitude) for the first M components
    
    modulus = np.abs(S[:M])

    # 4. Calculate Threshold T
    # T = sqrt(n * ln(1/0.05)) -> 95% threshold
    T = math.sqrt(n * math.log(1 / 0.05))

    # 5. Compute N1 (Observed number of peaks < T)
    # NIST expects 95% of the peaks to be below T.
    N1 = np.sum(modulus < T)
    
    # 6. Compute N0 (Theoretical expected number of peaks < T)
    # N0 = 0.95 * n / 2
    N0 = 0.95 * n / 2

    # 7. Calculate d statistic
    # d = (N1 - N0) / sqrt(n * 0.95 * 0.05 / 4)
    # The denominator is the standard deviation of the theoretical count.
    numerator = N1 - N0
    variance = (n * 0.95 * 0.05) / 4.0
    d = numerator / math.sqrt(variance)

    # 8. Calculate P-value
    # P-value = erfc(|d| / sqrt(2))
    P_value = erfc(abs(d) / math.sqrt(2))

    # 9. Verdict
    verdict = P_value >= alpha
    
    return P_value, verdict