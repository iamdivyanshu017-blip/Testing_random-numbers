import numpy as np
from typing import Tuple, Union

def calculate_excursion_statistic(data: np.ndarray) -> float:
    """
    Calculates the Excursion Statistic T as per NIST SP 800-90B.
    
    Formula:
        mean = sum(x) / L
        d_i = sum_{j=1..i} (x_j - mean)
        T = max(|d_1|, |d_2|, ..., |d_L|)
    
    Args:
        data (np.ndarray): The input sequence of numerical values.
        
    Returns:
        float: The maximum absolute deviation from the running average.
    """
    # 1. Calculate the mean of the sequence
    mean = np.mean(data)
    
    # 2. Calculate deviations (x_i - mean)
    deviations = data - mean
    
    # 3. Calculate the cumulative sum of deviations (running sum)
    # This is much faster than a loop in Python
    running_sums = np.cumsum(deviations)
    
    # 4. Find the maximum absolute value in the running sums
    statistic = np.max(np.abs(running_sums))
    
    return statistic

def run_iid_excursion_test(
    sample_data: Union[list, np.ndarray], 
    iterations: int = 10000, 
    verbose: bool = False
) -> Tuple[bool, int, float]:
    """
    Performs the full IID Excursion Test using permutation testing.
    
    Logic:
    1. Calculate Stat(original).
    2. Shuffle data 10,000 times, calculating Stat(permuted) each time.
    3. Rank original stat among permuted stats.
    4. Reject IID if Rank <= 5 or Rank >= 9995 (Rank starts at 1).
    
    Args:
        sample_data: The input entropy source data (list or numpy array).
        iterations: Number of permutations (Default 10,000 per NIST spec).
        verbose: If True, prints progress updates.

    Returns:
        (passed, rank, original_stat)
        - passed (bool): True if IID assumption holds, False if rejected.
        - rank (int): The position of original data among permutations (1 to N+1).
        - original_stat (float): The excursion statistic of the raw data.
    """
    # Ensure data is a float array for precision during mean calculation
    data = np.array(sample_data, dtype=np.float64)
    n = len(data)

    if n == 0:
        raise ValueError("Input data cannot be empty.")

    # 1. Calculate Statistic for Original Data
    original_stat = calculate_excursion_statistic(data)
    
    # 2. Run Permutations
    # We create a working copy to shuffle in-place for performance
    working_data = data.copy()
    perm_stats = np.zeros(iterations)
    
    for i in range(iterations):
        np.random.shuffle(working_data)
        perm_stats[i] = calculate_excursion_statistic(working_data)
        
        # Optional: Print progress every 10%
        if verbose and i % (iterations // 10) == 0:
            print(f"Permutations progress: {i}/{iterations}")

    # 3. Calculate Rank
    # Count how many permuted stats are strictly less than original stat
    # NIST 800-90B Section 5.1.1: We rank the combined set (10,000 perms + 1 original)
    # Rank is the number of vals < original + 1. 
    # (If equal, we break ties by assuming original is 'later', or average, 
    # but standard usually implies simple ranking).
    
    # Check strictly less
    count_less = np.sum(perm_stats < original_stat)
    
    # Check equal values (tie-breaking)
    # NIST approach generally assumes if C_0 equals some C_i, it falls within them.
    # A conservative strict rank is usually: count(x < val) + 1
    rank = count_less + 1
    
    # 4. Determine Pass/Fail (NIST Thresholds)
    # Reject if rank <= 5 or rank >= 9995 (for 10,000 iterations)
    # This represents a 0.1% significance level (0.001)
    
    passed = True
    if rank <= 5 or rank >= (iterations - 5):
        passed = False

    return passed, rank, original_stat

# --- Usage Example ---
if __name__ == "__main__":
    # Simulate some data
    # Case A: Random IID data (Should PASS)
    np.random.seed(42)
    iid_data = np.random.randint(0, 256, 1000) # 1000 bytes of random data
    
    print("Running Excursion Test on Random Data...")
    passed, rank, stat = run_iid_excursion_test(iid_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001 (Ideal range: 6-9994)")
    print(f"Statistic: {stat:.4f}\n")

    # Case B: Data with a drift/trend (Should FAIL)
    # Create data that slowly increases (non-stationary)
    drift_data = np.linspace(0, 100, 1000) + np.random.normal(0, 5, 1000)
    
    print("Running Excursion Test on Drifting Data...")
    passed, rank, stat = run_iid_excursion_test(drift_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001")
    print(f"Statistic: {stat:.4f}")