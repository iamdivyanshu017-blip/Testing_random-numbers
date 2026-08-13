import numpy as np
from typing import Tuple, Union

def calculate_max_run_length_binary(binary_data: np.ndarray) -> int:
    """
    Calculates the maximum length of a consecutive run of identical values
    in a binary sequence.
    
    Args:
        binary_data (np.ndarray): Array of 0s and 1s.
        
    Returns:
        int: The length of the longest run.
    """
    if len(binary_data) == 0:
        return 0
        
    # 1. Identify where values change (transitions)
    # binary_data[1:] != binary_data[:-1] returns a boolean array
    # True where a transition occurs (e.g., 0->1 or 1->0)
    transitions = binary_data[1:] != binary_data[:-1]
    
    # 2. Get indices of these transitions
    change_indices = np.where(transitions)[0]
    
    # 3. Calculate lengths
    # We add -1 (start boundary) and len-1 (end boundary) to the indices
    # to calculate the length of the first and last runs correctly.
    # Example: Runs end at indices 2 and 5. Boundaries: [-1, 2, 5, 9] (for len 10)
    # Lengths: 2-(-1)=3, 5-2=3, 9-5=4.
    boundaries = np.concatenate(([-1], change_indices, [len(binary_data) - 1]))
    
    run_lengths = np.diff(boundaries)
    
    if len(run_lengths) == 0:
        return 0 # Should not happen for non-empty input
        
    return int(np.max(run_lengths))

def run_len_runs_median_test(
    sample_data: Union[list, np.ndarray], 
    iterations: int = 10000,
    verbose: bool = False
) -> Tuple[bool, int, int]:
    """
    Performs the full IID Length of Runs Based on Median Test.
    
    Logic:
    1. Calculate Median (M).
    2. Map data to binary: 1 if x >= M, else 0.
    3. T = Length of the longest run in this binary sequence.
    4. Permutation test: Shuffle binary sequence, recalculate T.
    
    Args:
        sample_data: Input entropy source data.
        iterations: Number of permutations (Default 10,000).
        verbose: Print progress.

    Returns:
        (passed, rank, original_stat)
    """
    # Use float64 for precise median calculation
    data = np.array(sample_data, dtype=np.float64)
    
    if len(data) < 2:
        raise ValueError("Data length must be at least 2.")

    # 1. Calculate Median & Map to Binary (Pre-loop Optimization)
    median_val = np.median(data)
    
    # NIST Rule: y_i = 1 if x_i >= median, else 0
    # Using int8 saves memory for large datasets
    binary_seq = (data >= median_val).astype(np.int8)
    
    # 2. Calculate Statistic for Original Data
    original_stat = calculate_max_run_length_binary(binary_seq)
    
    # 3. Permutation Testing
    # We shuffle the binary sequence directly, saving massive overhead.
    working_seq = binary_seq.copy()
    perm_stats = np.zeros(iterations)
    
    for i in range(iterations):
        np.random.shuffle(working_seq)
        perm_stats[i] = calculate_max_run_length_binary(working_seq)
        
        if verbose and i % (iterations // 10) == 0:
            print(f"Permutations progress: {i}/{iterations}")

    # 4. Calculate Rank
    # Rank = count(perm_stats < original_stat) + 1
    count_less = np.sum(perm_stats < original_stat)
    rank = count_less + 1
    
    # 5. Determine Pass/Fail (NIST Thresholds)
    # Fail if Rank <= 5 or Rank >= 9995
    passed = True
    if rank <= 5 or rank >= (iterations - 5):
        passed = False
        
    return passed, rank, original_stat

# --- Usage Example ---
if __name__ == "__main__":
    # Case A: Random Data (Should Pass)
    # With random data, the "longest run" of heads or tails usually follows 
    # specific probabilities (related to log2(N)). It shouldn't be too long or too short.
    np.random.seed(42)
    random_data = np.random.randint(0, 256, 1000)
    
    print("Running Length of Runs (Median) Test on Random Data...")
    passed, rank, stat = run_len_runs_median_test(random_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001 (Ideal range: 6-9994)")
    print(f"Statistic (Max Run Length): {stat}\n")

    # Case B: Sticky/Biased Data (Should Fail - High Statistic)
    # Data that stays high for a long time, then low for a long time.
    # The binary mapping will look like: 11111111...00000000...
    # The max run length will be very large (e.g., 500).
    sticky_data = np.concatenate([np.ones(500) * 100, np.zeros(500)])
    
    print("Running Length of Runs (Median) Test on Sticky Data...")
    passed, rank, stat = run_len_runs_median_test(sticky_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001")
    print(f"Statistic (Max Run Length): {stat} (Expect very high rank)\n")
    
    # Case C: Rapid Oscillation (Should Fail - Low Statistic)
    # Data that strictly alternates 0, 100, 0, 100...
    # The binary map is 0, 1, 0, 1...
    # The max run length is exactly 1.
    # Random shuffles will almost certainly produce a run of 2 or 3 by chance,
    # so the original stat (1) will be smaller than almost all permutations.
    oscillating_data = np.array([0 if i % 2 == 0 else 100 for i in range(1000)])
    
    print("Running Length of Runs (Median) Test on Oscillating Data...")
    passed, rank, stat = run_len_runs_median_test(oscillating_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001")
    print(f"Statistic (Max Run Length): {stat} (Expect rank ~1)")