import numpy as np
from typing import Tuple, Union

def calculate_runs_on_binary(binary_data: np.ndarray) -> int:
    """
    Helper function to count runs in a binary sequence.
    
    A run is a sequence of identical values.
    Example: [0, 0, 1, 1, 1, 0] has 3 runs (00, 111, 0).
    
    Args:
        binary_data (np.ndarray): Boolean or Integer array of 0s and 1s.
        
    Returns:
        int: The number of runs.
    """
    if len(binary_data) == 0:
        return 0
        
    # Count transitions where current value != previous value
    # Total runs = transitions + 1
    transitions = np.sum(binary_data[1:] != binary_data[:-1])
    return int(transitions + 1)

def run_num_runs_median_test(
    sample_data: Union[list, np.ndarray], 
    iterations: int = 10000,
    verbose: bool = False
) -> Tuple[bool, int, int]:
    """
    Performs the full IID Number of Runs Based on Median Test.
    
    Logic:
    1. Calculate Median (M) of the dataset.
    2. Map data to binary: 1 if x >= M, else 0.
    3. T = Number of runs in this binary sequence.
    4. Permutation test: Shuffle the binary sequence and recalculate T.
    
    Args:
        sample_data: Input entropy source data.
        iterations: Number of permutations (Default 10,000).
        verbose: Print progress.

    Returns:
        (passed, rank, original_stat)
    """
    # Ensure data is float for accurate median calculation
    data = np.array(sample_data, dtype=np.float64)
    n = len(data)
    
    if n < 2:
        raise ValueError("Data length must be at least 2.")

    # 1. Calculate Median once
    median = np.median(data)
    
    # 2. Create Binary Mapping (The "Invariant")
    # NIST 800-90B Sec 5.1.5: If x_i >= median, y_i = 1; else y_i = 0
    # We use int8 for memory efficiency
    binary_seq = (data >= median).astype(np.int8)
    
    # 3. Calculate Statistic for Original Data
    original_stat = calculate_runs_on_binary(binary_seq)
    
    # 4. Permutation Testing
    # OPTIMIZATION: We shuffle the binary_seq directly.
    # Shuffling the original floats and re-calculating the median/mapping
    # would yield the EXACT same binary sequence (just in a different order).
    working_seq = binary_seq.copy()
    perm_stats = np.zeros(iterations)
    
    for i in range(iterations):
        np.random.shuffle(working_seq)
        perm_stats[i] = calculate_runs_on_binary(working_seq)
        
        if verbose and i % (iterations // 10) == 0:
            print(f"Permutations progress: {i}/{iterations}")

    # 5. Calculate Rank
    # Rank = count(perm_stats < original_stat) + 1
    count_less = np.sum(perm_stats < original_stat)
    rank = count_less + 1
    
    # 6. Determine Pass/Fail (NIST Thresholds)
    # Fail if Rank <= 5 or Rank >= 9995
    passed = True
    if rank <= 5 or rank >= (iterations - 5):
        passed = False
        
    return passed, rank, original_stat

# --- Usage Example ---
if __name__ == "__main__":
    # Case A: Random Data (Should Pass)
    # Random data should cross the median frequently but not too frequently.
    np.random.seed(42)
    random_data = np.random.randint(0, 256, 1000)
    
    print("Running Number of Runs (Median) Test on Random Data...")
    passed, rank, stat = run_num_runs_median_test(random_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001 (Ideal range: 6-9994)")
    print(f"Statistic (Runs): {stat}\n")

    # Case B: Clustered Data (Should Fail - Low Runs)
    # Data stays above median for a long time, then below for a long time.
    # Example: [0,0,0... 100,100,100...]
    # This simulates a low-frequency oscillation or drift.
    clustered_data = np.concatenate([np.zeros(500), np.ones(500) * 100])
    
    print("Running Number of Runs (Median) Test on Clustered Data...")
    passed, rank, stat = run_num_runs_median_test(clustered_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001")
    print(f"Statistic (Runs): {stat} (Expect very low rank, ~1)")

    # Case C: High Frequency Oscillation (Should Fail - High Runs)
    # Data jumps above and below median every single step.
    # Example: [0, 100, 0, 100, 0, 100...]
    oscillating_data = np.array([0 if i % 2 == 0 else 100 for i in range(1000)])
    
    print("\nRunning Number of Runs (Median) Test on Oscillating Data...")
    passed, rank, stat = run_num_runs_median_test(oscillating_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001")
    print(f"Statistic (Runs): {stat} (Expect very high rank)")