import numpy as np
import math
from scipy.special import erfc

def random_excursions_variant_test(bit_array: np.ndarray, alpha: float = 0.01) -> list:
    n = len(bit_array)
    states_to_test = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    if n == 0:
        return []

    # 1. Transform to Bipolar Sequence (-1, +1)
    X = np.where(bit_array == 0, -1, 1)
    
    # 2. Calculate Partial Sums (Random Walk) S_k
    S_k = np.concatenate(([0], np.cumsum(X), [0]))
    
    # 3. Identify Cycles (J)
    returns_to_zero = np.where(S_k == 0)[0]
    J = len(returns_to_zero) - 1 

    # --- UPDATED: HANDLE INSUFFICIENT CYCLES ---
    if J < 500:
        # Return a fail result for every state to maintain table structure
        return [
            (x, 0.0000, "SKIP (J < 500)") 
            for x in states_to_test
        ]
    
    # 4. Count Total Visits (V_x) to each state x
    # Extract the walk strictly within the cycles
    valid_walk = S_k[returns_to_zero[0]:returns_to_zero[-1]+1]
    unique, counts = np.unique(valid_walk, return_counts=True)
    visit_counts = dict(zip(unique, counts))
    
    results = []
    for x in states_to_test:
        V_x = visit_counts.get(x, 0)
        
        numerator = abs(V_x - J)
        denominator = math.sqrt(2 * J * (4 * abs(x) - 2))
        
        if denominator == 0:
            P_value = 0.0
        else:
            P_value = erfc(numerator / denominator)
        
        # Verdict is true (PASS) if P_value >= alpha
        verdict = P_value >= alpha
        results.append((x, round(P_value, 6), verdict))
        
    return results
