import numpy as np
from typing import Tuple, Union

def calculate_max_directional_run_length(data: np.ndarray) -> int:
    """
    Calculates the Length of Directional Runs statistic (T) per NIST SP 800-90B.
    
    Logic:
    1. Calculate differences: d_i = x_{i+1} - x_i.
    2. Discard d_i == 0.
    3. Convert to signs (+1, -1).
    4. Identify sequences of identical signs.
    5. T = The length of the LONGEST such sequence.
    
    Args:
        data (np.ndarray): The input sequence.
        
    Returns:
        int: The length of the longest consecutive directional run.
    """
    if len(data) < 2:
        return 0

    # 1. Calculate differences
    diffs = data[1:] - data[:-1]
    
    # 2. Filter out zeros
    # NIST requires ignoring equalities
    non_zero_diffs = diffs[diffs != 0]
    
    if len(non_zero_diffs) == 0:
        return 0
        
    # 3. Get signs
    signs = np.sign(non_zero_diffs)
    
    # 4. Vectorized Run Length Calculation
    # We need to find the lengths of all consecutive segments of identical values.
    # Logic: Find the indices where the value changes.
    # signs[1:] != signs[:-1] gives us a boolean array of change points.
    change_indices = np.where(signs[1:] != signs[:-1])[0]
    
    # We append the start (-1) and end (len-1) indices to calculate lengths correctly.
    # Example: If changes happen at index 1, the boundaries are start -> 1 -> end.
    # The diffs between these boundaries give the segment lengths.
    boundaries = np.concatenate(([-1], change_indices, [len(signs) - 1]))
    
    # Calculate lengths of every run
    run_lengths = np.diff(boundaries)
    
    # 5. Return the maximum length found
    return int(np.max(run_lengths))

def run_len_directional_runs_test(
    sample_data: Union[list, np.ndarray], 
    iterations: int = 10000,
    verbose: bool = False
) -> Tuple[bool, int, int]:
    """
    Performs the full IID Length of Directional Runs Test using permutation testing.
    
    Args:
        sample_data: Input entropy source data.
        iterations: Number of permutations (Default 10,000).
        verbose: Print progress.

    Returns:
        (passed, rank, original_stat)
    """
    data = np.array(sample_data, dtype=np.float64)
    
    if len(data) < 2:
        raise ValueError("Data length must be at least 2.")

    # 1. Calculate Statistic for Original Data
    original_stat = calculate_max_directional_run_length(data)
    
    # 2. Permutation Testing
    working_data = data.copy()
    perm_stats = np.zeros(iterations)
    
    for i in range(iterations):
        np.random.shuffle(working_data)
        perm_stats[i] = calculate_max_directional_run_length(working_data)
        
        if verbose and i % (iterations // 10) == 0:
            print(f"Permutations progress: {i}/{iterations}")

    # 3. Calculate Rank
    # Rank = count(perm_stats < original_stat) + 1
    count_less = np.sum(perm_stats < original_stat)
    rank = count_less + 1
    
    # 4. Determine Pass/Fail (NIST Thresholds)
    # Fail if Rank <= 5 or Rank >= 9995 (0.05% tails)
    passed = True
    if rank <= 5 or rank >= (iterations - 5):
        passed = False
        
    return passed, rank, original_stat

# --- Usage Example ---
if __name__ == "__main__":
    # Case A: Random Data (Should Pass)
    # Random data rarely has very long runs of pure increase or decrease
    np.random.seed(42)
    random_data = np.random.randint(0, 256, 1000)
    
    print("Running Length of Directional Runs Test on Random Data...")
    passed, rank, stat = run_len_directional_runs_test(random_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001 (Ideal range: 6-9994)")
    print(f"Statistic (Max Length): {stat}\n")

    # Case B: Monotonic Data (Should Fail - High Statistic)
    # If data is sorted, it is one massive run.
    # This represents extreme dependency (history determines future).
    monotonic_data = np.arange(1000)
    
    print("Running Length of Directional Runs Test on Monotonic Data...")
    passed, rank, stat = run_len_directional_runs_test(monotonic_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001")
    print(f"Statistic (Max Length): {stat} (Expect max length = N-1)")
    
    # Case C: Strictly Alternating Data (Should Fail - Low Statistic)
    # If data goes 0, 1, 0, 1... the max run length is just 1.
    # While "random" shuffling might produce short runs, having a max length of ONLY 1
    # is highly unlikely in a shuffled set of 1000 items.
    alternating_data = np.array([i % 2 for i in range(1000)])
    
    print("\nRunning Length of Directional Runs Test on Alternating Data...")
    passed, rank, stat = run_len_directional_runs_test(alternating_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001")
    print(f"Statistic (Max Length): {stat} (Expect rank ~1)")