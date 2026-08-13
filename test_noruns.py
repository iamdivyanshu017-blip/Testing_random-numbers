import numpy as np
from typing import Tuple, Union

def calculate_num_directional_runs(data: np.ndarray) -> int:
    """
    Calculates the Number of Directional Runs statistic (T) per NIST SP 800-90B.
    
    Logic:
    1. Calculate differences between adjacent items: d_i = x_{i+1} - x_i.
    2. Discard any d_i == 0 (equal values do not interrupt or form runs).
    3. Map remaining d_i to signs (+1 or -1).
    4. A "run" is a sequence of identical signs.
    5. T = Number of such runs.
    
    Args:
        data (np.ndarray): The input sequence of numerical values.
        
    Returns:
        int: The number of directional runs. Returns 0 if all values are identical.
    """
    if len(data) < 2:
        return 0

    # 1. Calculate differences (x_{i+1} - x_i)
    # Vectorized operation is much faster than looping
    diffs = data[1:] - data[:-1]
    
    # 2. Filter out zeros (where x_{i+1} == x_i)
    # NIST 800-90B requires ignoring equalities for directional runs
    non_zero_diffs = diffs[diffs != 0]
    
    if len(non_zero_diffs) == 0:
        return 0
        
    # 3. Get signs (-1 or +1)
    signs = np.sign(non_zero_diffs)
    
    # 4. Count runs
    # A new run starts whenever the sign changes compared to the previous one.
    # We compare the array shifted by one against itself.
    # signs[1:] != signs[:-1] creates a boolean array of transitions.
    # Summing this gives the number of changes.
    # We add 1 because the first sequence counts as the first run.
    num_runs = 1 + np.sum(signs[1:] != signs[:-1])
    
    return int(num_runs)

def run_directional_runs_test(
    sample_data: Union[list, np.ndarray], 
    iterations: int = 10000,
    verbose: bool = False
) -> Tuple[bool, int, int]:
    """
    Performs the full IID Number of Directional Runs Test using permutation testing.
    
    Args:
        sample_data: Input entropy source data.
        iterations: Number of permutations (Default 10,000).
        verbose: Print progress.

    Returns:
        (passed, rank, original_stat)
    """
    # Cast to float or int usually doesn't matter for comparison, 
    # but int is safer for exact equality checks if inputs are ints.
    data = np.array(sample_data, dtype=np.float64)
    
    if len(data) < 2:
        raise ValueError("Data length must be at least 2.")

    # 1. Calculate Statistic for Original Data
    original_stat = calculate_num_directional_runs(data)
    
    # 2. Permutation Testing
    working_data = data.copy()
    perm_stats = np.zeros(iterations)
    
    for i in range(iterations):
        np.random.shuffle(working_data)
        perm_stats[i] = calculate_num_directional_runs(working_data)
        
        if verbose and i % (iterations // 10) == 0:
            print(f"Permutations progress: {i}/{iterations}")

    # 3. Calculate Rank
    # Rank is the position of the original stat in the sorted list of permuted stats
    # Specifically: Number of permuted stats strictly less than original + 1
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
    # Expect runs to be roughly N/2 + N/2... random mix
    np.random.seed(100)
    random_data = np.random.randint(0, 256, 1000)
    
    print("Running Directional Runs Test on Random Data...")
    passed, rank, stat = run_directional_runs_test(random_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001 (Ideal range: 6-9994)")
    print(f"Statistic (Runs): {stat}\n")

    # Case B: Sawtooth Pattern (Should Fail - High Runs)
    # Data goes up, down, up, down... maximize runs
    # Example: 0, 10, 0, 10, 0, 10...
    sawtooth = np.array([10 if i % 2 == 0 else 0 for i in range(1000)])
    
    print("Running Directional Runs Test on Sawtooth Data...")
    passed, rank, stat = run_directional_runs_test(sawtooth)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001")
    print(f"Statistic (Runs): {stat} (Expect very high rank)")
    
    # Case C: Monotonic Data (Should Fail - Low Runs)
    # Data just goes up. 1 run.
    monotonic = np.arange(1000)
    
    print("\nRunning Directional Runs Test on Monotonic Data...")
    passed, rank, stat = run_directional_runs_test(monotonic)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001")
    print(f"Statistic (Runs): {stat} (Expect rank ~1)")