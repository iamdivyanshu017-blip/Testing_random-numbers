import numpy as np
from scipy.stats import chi2

def berlekamp_massey(bit_array: np.ndarray) -> int:
    n = len(bit_array)
    b = np.zeros(n, dtype=int)
    c = np.zeros(n, dtype=int)
    b[0] = 1
    c[0] = 1
    l = 0
    m = -1
    for N in range(n):
        d = 0
        for i in range(l + 1):
            d ^= c[i] & bit_array[N - i]
        if d == 1:
            t = c.copy()
            p = N - m
            if l * 2 <= N:
                if p < n:
                    shift = np.zeros(n, dtype=int)
                    shift[p:] = b[:n-p]
                    c ^= shift
                l = N + 1 - l
                b = t
                m = N
            else:
                 shift = np.zeros(n, dtype=int)
                 shift[p:] = b[:n-p]
                 c ^= shift
    return l

def linear_complexity_test(bit_array: np.ndarray, M: int = 500, alpha: float = 0.01):
    n = len(bit_array)
    N = int(n // M)
    if N < 3: return 0.0, False 
    
    pi = np.array([0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833])
    mu = M / 2.0 + (9.0 + (-1)**(M + 1)) / 36.0 - ((M / 3.0) + (2.0 / 9.0)) / (2**M)
    
    v = np.zeros(7)
    for i in range(N):
        block = bit_array[i*M : (i+1)*M]
        L_i = berlekamp_massey(block)
        T = (-1)**M * (L_i - mu) + 2.0 / 9.0
        
        if T <= -2.5: v[0] += 1
        elif T <= -1.5: v[1] += 1
        elif T <= -0.5: v[2] += 1
        elif T <= 0.5: v[3] += 1
        elif T <= 1.5: v[4] += 1
        elif T <= 2.5: v[5] += 1
        else:         v[6] += 1
        
    chi_sq = np.sum((v - N * pi)**2 / (N * pi))
    p_value = chi2.sf(chi_sq, 6)
    
    return p_value, p_value >= alpha