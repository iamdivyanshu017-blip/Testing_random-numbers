import numpy as np
from typing import Tuple, Union, List

def calculate_covariance_statistic(data: np.ndarray, lag: int) -> float:
    """
    Calculates the Covariance Statistic (T) for a specific lag p.
    
    Logic:
    T = Sum(data[i] * data[i+p])
    We use the simplified cross-product sum because the mean is invariant
    under permutation, making the full covariance formula unnecessary for ranking.
    
    Args:
        data (np.ndarray): Input sequence (float64 recommended).
        lag (int): The lag distance.
        
    Returns:
        float: The sum of products.
    """
    n = len(data)
    if n <= lag:
        return 0.0
        
    # Vectorized calculation:
    # Multiply array by its shifted version and sum the result.
    # data[:-lag] is indices 0 to N-lag-1
    # data[lag:]  is indices lag to N-1
    product_sum = np.sum(data[:-lag] * data[lag:])
    
    return float(product_sum)

def run_covariance_test(
    sample_data: Union[list, np.ndarray], 
    iterations: int = 10000, 
    verbose: bool = False
) -> Tuple[bool, List[int], List[float]]:
    """
    Performs the full IID Covariance Test for all NIST-specified lags.
    
    Lags tested: 1, 2, 8, 16, 32.
    
    Args:
        sample_data: Input entropy source data.
        iterations: Number of permutations (Default 10,000).
        verbose: Print progress.

    Returns:
        (passed, ranks, original_stats)
        - passed (bool): True if ALL lags pass.
        - ranks (List[int]): The rank for each lag.
        - original_stats (List[float]): The statistic T for each lag.
    """
    # 1. Prepare Data
    # Use float64 to avoid overflow when summing products of integers
    data = np.array(sample_data, dtype=np.float64)
    n = len(data)
    
    # NIST requires specific lags
    lags = [1, 2, 8, 16, 32]
    
    if n <= max(lags):
        raise ValueError(f"Data length ({n}) must be greater than the max lag (32).")

    # 2. Calculate Original Statistics
    original_stats = []
    for p in lags:
        original_stats.append(calculate_covariance_statistic(data, p))
    
    # 3. Permutation Testing
    # Track "less than" counts for each lag independently
    counts_less = np.zeros(len(lags), dtype=int)
    
    working_data = data.copy()
    
    for i in range(iterations):
        np.random.shuffle(working_data)
        
        for idx, p in enumerate(lags):
            perm_stat = calculate_covariance_statistic(working_data, p)
            
            if perm_stat < original_stats[idx]:
                counts_less[idx] += 1
        
        if verbose and i % (iterations // 10) == 0:
            print(f"Permutations progress: {i}/{iterations}")

    # 4. Calculate Ranks and Determine Pass/Fail
    ranks = counts_less + 1
    passed = True
    
    # Check all ranks against thresholds
    for r in ranks:
        if r <= 5 or r >= (iterations - 5):
            passed = False
            
    return passed, ranks.tolist(), original_stats

# --- Usage Example ---
if __name__ == "__main__":
    # Case A: Random Data (Should Pass)
    # Correlation between adjacent samples should be near zero (relative to shuffled).
    np.random.seed(42)
    random_data = np.random.randint(0, 256, 1000)
    
    print("Running Covariance Test on Random Data...")
    passed, ranks, stats = run_covariance_test(random_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Lags: [1, 2, 8, 16, 32]")
    print(f"Ranks: {ranks} (Ideal range: 6-9994)")
    print(f"Stats: {stats}\n")

    # Case B: Correlated Data (Should Fail)
    # Create data where x[i] is strongly related to x[i-1].
    # A simple "random walk" or integrated noise.
    # x_i = x_{i-1} + noise
    walk_data = np.zeros(1000)
    for i in range(1, 1000):
        walk_data[i] = walk_data[i-1] + np.random.normal(0, 1)
        
    print("Running Covariance Test on Random Walk Data...")
    passed, ranks, stats = run_covariance_test(walk_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Lags: [1, 2, 8, 16, 32]")
    print(f"Ranks: {ranks}")
    print(f"Stats: {stats}")
    # Expectation: 
    # High positive correlation at Lag 1 means the product sum will be maximized.
    # Rank should be very high (>9995) for Lag 1, and likely high for others (2, 8...) 
    # as the correlation decays slowly.
    
    # Case C: Anti-Correlated Data (Should Fail - Low Rank)
    # x_i = -x_{i-1}
    # Lag 1 products will be negative, minimizing the sum.
    # Shuffling destroys this structure, making the sum closer to 0 (larger than original).
    anti_corr = np.array([10 if i%2==0 else -10 for i in range(1000)])
    
    print("\nRunning Covariance Test on Anti-Correlated Data...")
    passed, ranks, stats = run_covariance_test(anti_corr)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank (Lag 1): {ranks[0]}")
    # Expectation: Rank < 5 for Lag 1.