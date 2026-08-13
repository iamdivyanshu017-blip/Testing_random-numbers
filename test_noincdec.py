import numpy as np
from typing import Tuple, Union

def calculate_max_increases_decreases(data: np.ndarray) -> int:
    """
    Calculates the Number of Increases and Decreases statistic (T) per NIST SP 800-90B.
    
    Logic:
    1. Calculate differences: d_i = x_{i+1} - x_i.
    2. Discard d_i == 0.
    3. Count N_up (where d_i > 0).
    4. Count N_down (where d_i < 0).
    5. T = max(N_up, N_down).
    
    Args:
        data (np.ndarray): The input sequence.
        
    Returns:
        int: The maximum of the increase count or decrease count.
    """
    if len(data) < 2:
        return 0

    # 1. Calculate differences (Vectorized)
    diffs = data[1:] - data[:-1]
    
    # 2. Filter out zeros (equalities do not count)
    non_zero_diffs = diffs[diffs != 0]
    
    n_total = len(non_zero_diffs)
    if n_total == 0:
        return 0
        
    # 3. Count Increases
    # np.sum on a boolean array counts the Trues
    n_up = np.sum(non_zero_diffs > 0)
    
    # 4. Count Decreases
    # Since we filtered zeros, n_down is just the remainder
    n_down = n_total - n_up
    
    # 5. Return Max
    return int(max(n_up, n_down))

def run_increases_decreases_test(
    sample_data: Union[list, np.ndarray], 
    iterations: int = 10000,
    verbose: bool = False
) -> Tuple[bool, int, int]:
    """
    Performs the full IID Number of Increases and Decreases Test.
    
    Args:
        sample_data: Input entropy source data.
        iterations: Number of permutations (Default 10,000).
        verbose: Print progress.

    Returns:
        (passed, rank, original_stat)
    """
    # Ensure data is float/int array
    data = np.array(sample_data, dtype=np.float64)
    
    if len(data) < 2:
        raise ValueError("Data length must be at least 2.")

    # 1. Calculate Statistic for Original Data
    original_stat = calculate_max_increases_decreases(data)
    
    # 2. Permutation Testing
    working_data = data.copy()
    perm_stats = np.zeros(iterations)
    
    for i in range(iterations):
        np.random.shuffle(working_data)
        perm_stats[i] = calculate_max_increases_decreases(working_data)
        
        if verbose and i % (iterations // 10) == 0:
            print(f"Permutations progress: {i}/{iterations}")

    # 3. Calculate Rank
    # Rank = count(perm_stats < original_stat) + 1
    count_less = np.sum(perm_stats < original_stat)
    rank = count_less + 1
    
    # 4. Determine Pass/Fail (NIST Thresholds)
    # Fail if Rank <= 5 or Rank >= 9995
    passed = True
    if rank <= 5 or rank >= (iterations - 5):
        passed = False
        
    return passed, rank, original_stat

# --- Usage Example ---
if __name__ == "__main__":
    # Case A: Random Data (Should Pass)
    # Increases and Decreases should be roughly equal (approx N/2 each).
    # The max(N_up, N_down) will be slightly above N/2.
    np.random.seed(42)
    random_data = np.random.randint(0, 256, 1000)
    
    print("Running Increases/Decreases Test on Random Data...")
    passed, rank, stat = run_increases_decreases_test(random_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001 (Ideal range: 6-9994)")
    print(f"Statistic: {stat} (Approx 500 for N=1000)\n")

    # Case B: Strictly Increasing Data (Should Fail)
    # Here N_up = 999, N_down = 0. Statistic = 999.
    # Random shuffles will mix ups and downs, lowering the max.
    # Rank should be extremely high (original is higher than all shuffles).
    increasing_data = np.arange(1000)
    
    print("Running Increases/Decreases Test on Increasing Data...")
    passed, rank, stat = run_increases_decreases_test(increasing_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001")
    print(f"Statistic: {stat} (Expect max possible value)\n")
    
    # Case C: Biased Data (Subtle Fail)
    # Data that tends to go up 2 steps and down 1 step.
    # This creates an imbalance between N_up and N_down.
    # While shuffles might look similar, if the magnitude of steps matters, 
    # this test focuses only on the COUNT of steps.
    # Note: If the values are just permuted, the number of ups/downs in the
    # shuffled set is purely determined by the ordering.
    # If the dataset has many unique values, a sorted version maximizes N_up.
    # A perfectly alternating version (0,10,0,10) balances N_up and N_down (N/2).
    # This test detects if the original order has an unusually high imbalance.