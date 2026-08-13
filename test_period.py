import numpy as np
from typing import Tuple, Union, List

def calculate_periodicity_statistic(data: np.ndarray, lag: int) -> int:
    """
    Calculates the Periodicity Statistic (T) for a specific lag p.
    
    Logic:
    Count occurrences where x_i == x_{i+p}.
    
    Args:
        data (np.ndarray): Input sequence.
        lag (int): The distance to check for equality.
        
    Returns:
        int: The number of matches found at this lag.
    """
    n = len(data)
    if n <= lag:
        return 0
        
    # Vectorized comparison:
    # Compare data[0...N-p] with data[p...N]
    matches = (data[:-lag] == data[lag:])
    
    # Sum the boolean array (True=1, False=0)
    return int(np.sum(matches))

def run_periodicity_test(
    sample_data: Union[list, np.ndarray], 
    iterations: int = 10000, 
    verbose: bool = False
) -> Tuple[bool, List[int], List[int]]:
    """
    Performs the full IID Periodicity Test for all NIST-specified lags.
    
    Lags tested: 1, 2, 8, 16, 32.
    
    Args:
        sample_data: Input entropy source data.
        iterations: Number of permutations (Default 10,000).
        verbose: Print progress.

    Returns:
        (passed, ranks, original_stats)
        - passed (bool): True if ALL lags pass.
        - ranks (List[int]): The rank for each lag [p=1, p=2, ..., p=32].
        - original_stats (List[int]): The statistic T for each lag.
    """
    # 1. Prepare Data
    data = np.array(sample_data)
    n = len(data)
    
    # NIST requires specific lags
    lags = [1, 2, 8, 16, 32]
    
    # Validate data length
    if n <= max(lags):
        raise ValueError(f"Data length ({n}) must be greater than the max lag (32).")

    # 2. Calculate Original Statistics for all lags
    original_stats = []
    for p in lags:
        original_stats.append(calculate_periodicity_statistic(data, p))
    
    # 3. Permutation Testing
    # We maintain a separate counter for each lag
    # counts_less[0] tracks lag=1, counts_less[1] tracks lag=2, etc.
    counts_less = np.zeros(len(lags), dtype=int)
    
    working_data = data.copy()
    
    for i in range(iterations):
        np.random.shuffle(working_data)
        
        # For each shuffled set, calculate T for all 5 lags
        for idx, p in enumerate(lags):
            perm_stat = calculate_periodicity_statistic(working_data, p)
            
            if perm_stat < original_stats[idx]:
                counts_less[idx] += 1
        
        if verbose and i % (iterations // 10) == 0:
            print(f"Permutations progress: {i}/{iterations}")

    # 4. Calculate Ranks and Determine Pass/Fail
    ranks = counts_less + 1
    passed = True
    
    # Determine pass/fail for EACH lag
    # If ANY lag fails, the whole Periodicity Test fails
    for r in ranks:
        if r <= 5 or r >= (iterations - 5):
            passed = False
            # We don't break here so we can return all ranks for debugging
            
    return passed, ranks.tolist(), original_stats

# --- Usage Example ---
if __name__ == "__main__":
    # Case A: Random Data (Should Pass)
    # With random bytes, chance of collision is 1/256.
    # Expected matches ~ N/256.
    np.random.seed(42)
    random_data = np.random.randint(0, 256, 1000)
    
    print("Running Periodicity Test on Random Data...")
    passed, ranks, stats = run_periodicity_test(random_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Lags: [1, 2, 8, 16, 32]")
    print(f"Ranks: {ranks} (Ideal range: 6-9994)")
    print(f"Stats: {stats}\n")

    # Case B: Periodic Data (Should Fail at specific lag)
    # Create a pattern that repeats every 8 samples: 0,1,2,3,4,5,6,7, 0,1...
    # This should trigger a fail specifically at lag=8 (and multiples like 16, 32).
    periodic_data = np.array([i % 8 for i in range(1000)])
    
    print("Running Periodicity Test on Periodic (Lag 8) Data...")
    passed, ranks, stats = run_periodicity_test(periodic_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Lags: [1, 2, 8, 16, 32]")
    print(f"Ranks: {ranks}")
    print(f"Stats: {stats}")
    # Expectation: 
    # Rank for Lag 1, 2 might be normal or low.
    # Rank for Lag 8 should be >9995 (Massive number of matches compared to random shuffle).
    # Rank for Lag 16, 32 should also be >9995 (Harmonics).