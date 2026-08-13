import numpy as np
from typing import Tuple, Union

def calculate_max_collision_statistic(data: np.ndarray) -> int:
    """
    Calculates the Maximum Collision Statistic (T) per NIST SP 800-90B.
    
    Logic:
    1. Iterate through data, adding unique values to a current set.
    2. Upon finding a duplicate (collision), record the count (segment length).
    3. Clear the set and restart counting.
    4. T = The maximum segment length observed.
    
    Args:
        data (np.ndarray): Input sequence.
        
    Returns:
        int: The maximum number of samples in a collision-free segment.
    """
    n = len(data)
    if n == 0:
        return 0

    max_run_length = 0
    current_run_len = 0
    
    # Use a set for O(1) average time complexity lookups
    seen = set()
    
    for value in data:
        # Check if value is already in the current segment
        if value in seen:
            # Collision occurred!
            # The run length includes the samples BEFORE this collision.
            # According to NIST 800-90B logic: "until a collision is observed...
            # The next segment begins with the sample following the collision."
            # The statistic is the number of samples in the segment.
            # Since we incremented current_run_len for every unique add,
            # we record that, then add 1 for the collision sample itself?
            # Standard interpretation: The "collision interval" counts the samples 
            # processed *including* the one that caused the collision.
            current_run_len += 1 
            
            if current_run_len > max_run_length:
                max_run_length = current_run_len
            
            # Reset for next segment
            seen.clear()
            current_run_len = 0
        else:
            seen.add(value)
            current_run_len += 1
            
    # Check the final partial segment
    # NIST SP 800-90B typically discards the last segment if it doesn't end in a collision.
    # However, for "Maximum" logic, if the data ends with a massive unique stream,
    # discarding it might hide non-random behavior. 
    # Strict NIST implementation: Discard the final partial segment.
    
    # If NO collisions occurred at all, max_run_length is 0.
    # In that case, the entire dataset is one unique run. 
    if max_run_length == 0 and current_run_len > 0:
        # Corner case: The dataset was fully unique (no collisions ever).
        # We report the full length.
        max_run_length = current_run_len

    return int(max_run_length)

def run_max_collision_test(
    sample_data: Union[list, np.ndarray], 
    iterations: int = 10000, 
    verbose: bool = False
) -> Tuple[bool, int, int]:
    """
    Performs the full IID Maximum Collision Test.
    
    Args:
        sample_data: Input entropy source data.
        iterations: Number of permutations (Default 10,000).
        verbose: Print progress.

    Returns:
        (passed, rank, original_stat)
    """
    # 1. Prepare Data
    data = np.array(sample_data)
    
    # 2. Calculate Statistic for Original Data
    original_stat = calculate_max_collision_statistic(data)
    
    # 3. Permutation Testing
    working_data = data.copy()
    perm_stats = np.zeros(iterations)
    
    for i in range(iterations):
        np.random.shuffle(working_data)
        perm_stats[i] = calculate_max_collision_statistic(working_data)
        
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
    # With bytes (0-255), we expect collisions fairly often (birthday paradox).
    # The max run shouldn't deviate wildly from shuffled versions.
    np.random.seed(42)
    random_data = np.random.randint(0, 256, 1000)
    
    print("Running Maximum Collision Test on Random Data...")
    passed, rank, stat = run_max_collision_test(random_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001 (Ideal range: 6-9994)")
    print(f"Statistic (Max Collision Interval): {stat}\n")

    # Case B: Repeating Pattern (Should Fail - High Statistic)
    # A perfect cycle: 0, 1, 2, ... 255, 0, 1, ...
    # This maximizes the time between collisions (256 samples every time).
    # Random shuffling will likely bring two same values closer together.
    # Therefore, the original stat (256) will be higher than most shuffled stats.
    pattern_data = np.array([i % 256 for i in range(2000)])
    
    print("Running Maximum Collision Test on Repeating Pattern...")
    passed, rank, stat = run_max_collision_test(pattern_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001")
    print(f"Statistic (Max Collision Interval): {stat} (Expect rank > 9995)")