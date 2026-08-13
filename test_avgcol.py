import numpy as np
from typing import Tuple, Union

def calculate_average_collision_statistic(data: np.ndarray) -> float:
    """
    Calculates the Average Collision Statistic (T) per NIST SP 800-90B.
    
    Logic:
    1. Start a segment.
    2. Add values to a set until a duplicate is found.
    3. The number of samples in this segment (including the duplicate) is the run length.
    4. Clear the set and repeat.
    5. T = Average of these run lengths.
    
    Args:
        data (np.ndarray): Input sequence.
        
    Returns:
        float: The average number of samples to find a collision.
    """
    n = len(data)
    if n == 0:
        return 0.0

    run_lengths = []
    
    # Use a set for O(1) lookups
    # In production, clearing a large set can be costly. 
    # For integer data (0-255), a boolean array is faster than a set.
    # However, for general inputs (arbitrary floats/ints), a set is safer.
    seen = set()
    current_run_len = 0
    
    for value in data:
        current_run_len += 1
        
        if value in seen:
            # Collision found!
            run_lengths.append(current_run_len)
            # Reset for next segment
            seen.clear()
            # The current value starts the new segment (implied by spec logic,
            # actually spec says "discard the samples" and restart.
            # Usually implies the collision value is consumed.
            # NIST Spec: "j segments... run lengths L_j"
            # We clear seen, and the loop continues.
            # BUT: We need to ensure the *current* value is added to the *new* set?
            # 800-90B Logic: "Until a collision is observed... The next segment begins 
            # with the sample *following* the collision."
            # So the current 'value' ends the previous run. 
            # We clear seen. We set current_run_len = 0.
            current_run_len = 0
        else:
            seen.add(value)
            
    # NIST Note: If the data ends without a final collision, the last partial segment 
    # is DISCARDED (it's not a complete measurement of time-to-collision).
    
    if len(run_lengths) == 0:
        # Corner case: No collisions found in the entire dataset (all unique)
        # This returns 0 or the full length depending on interpretation, 
        # but technically undefined. We return 0.0 to signal strict uniqueness.
        return 0.0
        
    return float(np.mean(run_lengths))

def run_avg_collision_test(
    sample_data: Union[list, np.ndarray], 
    iterations: int = 10000,
    verbose: bool = False
) -> Tuple[bool, int, float]:
    """
    Performs the full IID Average Collision Test.
    
    Args:
        sample_data: Input entropy source data.
        iterations: Number of permutations (Default 10,000).
        verbose: Print progress.

    Returns:
        (passed, rank, original_stat)
    """
    # 1. Prepare Data
    # Collisions rely on exact equality. 
    # If inputs are floats, precision issues might mask collisions.
    # NIST usually assumes integer symbols or discretized data.
    data = np.array(sample_data)
    
    # 2. Calculate Statistic for Original Data
    original_stat = calculate_average_collision_statistic(data)
    
    # 3. Permutation Testing
    # Optimization Note: Shuffling creates new orders, changing collision times.
    # Logic remains O(N) per iteration. Total O(M*N).
    # For Python, this loop is the bottleneck.
    # To optimize, we can use a pre-allocated boolean array if values are small integers (0-255).
    # If values are large/unbounded, we stick to sets.
    
    working_data = data.copy()
    perm_stats = np.zeros(iterations)
    
    # Check if we can apply small-integer optimization (0-255 range)
    is_byte_data = False
    if np.issubdtype(data.dtype, np.integer):
        if np.min(data) >= 0 and np.max(data) <= 255:
            is_byte_data = True

    # Pre-compile the inner function if using Numba for speed (optional but recommended for production)
    # Here we stick to pure NumPy/Python for compatibility.
    
    for i in range(iterations):
        np.random.shuffle(working_data)
        
        # We inline the calculation for speed if possible, or just call the function.
        # Calling function is cleaner.
        perm_stats[i] = calculate_average_collision_statistic(working_data)
        
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
    # Case A: Random Byte Data (Should Pass)
    # For bytes (0-255), avg collision time is roughly sqrt(2*256) ≈ 22 by birthday paradox,
    # but since we clear the set, it's slightly different.
    np.random.seed(42)
    random_data = np.random.randint(0, 256, 1000)
    
    print("Running Average Collision Test on Random Data...")
    passed, rank, stat = run_avg_collision_test(random_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001 (Ideal range: 6-9994)")
    print(f"Statistic (Avg Samples to Collision): {stat:.4f}\n")

    # Case B: Limited Alphabet (Should Fail - Low Statistic)
    # If data only uses values 0 and 1, collision happens almost immediately (avg ~2 or 3).
    # Random shuffling of 0s and 1s won't change the statistic much IF the ratio is fixed.
    # WAIT: Permutation testing preserves the frequency count.
    # If the original data is [0, 1, 0, 1...] vs [0, 0, 1, 1...], 
    # [0, 1, 0, 1] -> collides at 3 (0,1,0).
    # [0, 0, 1, 1] -> collides at 2 (0,0).
    # So order DOES matter.
    
    # Let's try repeating pattern vs random shuffle of same elements.
    # Pattern: 0, 1, 2, 3, 4, 0, 1, 2, 3, 4... (Collision every 6 samples)
    pattern_data = np.array([i % 5 for i in range(1000)])
    
    print("Running Average Collision Test on Repeating Pattern...")
    passed, rank, stat = run_avg_collision_test(pattern_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001")
    print(f"Statistic: {stat:.4f} (Expect High Rank - collision delayed)")
