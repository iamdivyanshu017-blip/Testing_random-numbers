import numpy as np
from scipy.stats import chi2

def random_excursions_test(bit_array: np.ndarray, alpha: float = 0.01) -> list:
    n = len(bit_array)
    if n == 0: return []

    # 1. Prepare Random Walk
    X = np.where(bit_array == 0, -1, 1)
    S_k = np.concatenate(([0], np.cumsum(X), [0]))
    
    # 2. Cycle Definition (J)
    zeros = np.where(S_k == 0)[0]
    J = len(zeros) - 1
    
    # NIST Requirement: J >= 500
    if J < 500:
        # User Request: Return empty if insufficient cycles (Leave blank)
        return []

    # 3. Theoretical Probabilities (NIST SP 800-22 Table)
    state_probs = {
        1: [0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0313],
        2: [0.7500, 0.0625, 0.0469, 0.0352, 0.0264, 0.0790],
        3: [0.8333, 0.0278, 0.0231, 0.0193, 0.0161, 0.0804],
        4: [0.8750, 0.0156, 0.0137, 0.0119, 0.0104, 0.0734]
    }
    
    states_to_test = [-4, -3, -2, -1, 1, 2, 3, 4]
    results = []

    for x in states_to_test:
        pi = np.array(state_probs[abs(x)])
        freq_counters = np.zeros(6) # k=0 to >=5
        
        for i in range(J):
            cycle = S_k[zeros[i]+1 : zeros[i+1]]
            count = np.sum(cycle == x)
            if count >= 5: freq_counters[5] += 1
            else: freq_counters[count] += 1
        
        expected = J * pi
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            terms = (freq_counters - expected)**2 / expected
            chi_sq = np.sum(np.nan_to_num(terms))
        
        p_val = chi2.sf(chi_sq, 5) # df=5
        verdict = p_val >= alpha
        results.append((x, p_val, verdict))

    return results