import numpy as np
from typing import Tuple, Union

def calculate_compression_statistic(data: np.ndarray) -> int:
    """
    Calculates the Compression Statistic (T) using the NIST LZ78Y algorithm.
    
    Logic:
    1. Init dictionary with all unique symbols in data.
    2. Read symbols. Maintain a current 'prefix'.
    3. If 'prefix + symbol' is in dictionary:
       Extend prefix.
    4. If not in dictionary:
       Add 'prefix + symbol' to dictionary.
       Increment statistic (phrase count).
       Reset prefix to empty.
    5. If dictionary grows > 65536, reset it to initial state.
    
    Args:
        data (np.ndarray): Input sequence.
        
    Returns:
        int: The number of new phrases added to the dictionary.
    """
    n = len(data)
    if n == 0:
        return 0

    # 1. Initialize Dictionary
    # The dictionary maps a byte-sequence (tuple) to existence (True).
    # We use a set for O(1) lookups.
    # We must convert numpy scalars to standard Python types for efficient hashing in sets.
    # (e.g., np.int64 is slower to hash than int).
    
    # Identify unique values for initialization
    distinct_values = sorted(list(set(data)))
    
    # Determine the initial dictionary state
    # Dictionary contains tuples of length 1: {(val1,), (val2,), ...}
    initial_dict = set((x,) for x in distinct_values)
    
    # Working dictionary
    dictionary = initial_dict.copy()
    
    # Current prefix (tuple)
    prefix = ()
    
    statistic = 0
    max_dict_size = 65536
    
    for x in data:
        # Create candidate phrase: prefix + current symbol
        # Note: (x,) creates a single-element tuple
        candidate = prefix + (x,)
        
        if candidate in dictionary:
            # Pattern exists, extend the prefix
            prefix = candidate
        else:
            # Pattern is new
            # 1. Add to dictionary
            dictionary.add(candidate)
            statistic += 1
            
            # 2. Reset prefix (start new phrase scan)
            prefix = ()
            
            # 3. Check Dictionary Size Limit
            if len(dictionary) >= max_dict_size:
                dictionary = initial_dict.copy()
                prefix = ()

    return statistic

def run_compression_test(
    sample_data: Union[list, np.ndarray], 
    iterations: int = 10000, 
    verbose: bool = False
) -> Tuple[bool, int, int]:
    """
    Performs the full IID Compression Test.
    
    Args:
        sample_data: Input entropy source data.
        iterations: Number of permutations (Default 10,000).
        verbose: Print progress.

    Returns:
        (passed, rank, original_stat)
    """
    # Use standard python ints/floats for set hashing efficiency
    data_np = np.array(sample_data)
    # Convert to Python list of standard types (int or float) to speed up tuple creation
    # inside the inner loop. Numpy scalars in tuples are slower.
    if np.issubdtype(data_np.dtype, np.integer):
        data = [int(x) for x in data_np]
    else:
        data = [float(x) for x in data_np]

    # 1. Calculate Statistic for Original Data
    original_stat = calculate_compression_statistic(data)
    
    # 2. Permutation Testing
    perm_stats = np.zeros(iterations)
    
    # Working copy for shuffling
    # We work with the numpy array for fast shuffling, convert to list for processing
    working_data_np = data_np.copy()
    
    for i in range(iterations):
        np.random.shuffle(working_data_np)
        
        # Convert to list for the compression function (faster hashing)
        # Note: For very large N, this conversion cost might outweigh hashing benefits,
        # but for N=1000 (typical IID), it's faster.
        if np.issubdtype(working_data_np.dtype, np.integer):
            perm_list = [int(x) for x in working_data_np]
        else:
            perm_list = [float(x) for x in working_data_np]
            
        perm_stats[i] = calculate_compression_statistic(perm_list)
        
        if verbose and i % (iterations // 10) == 0:
            print(f"Permutations progress: {i}/{iterations}")

    # 3. Calculate Rank
    # Rank = count(perm_stats < original_stat) + 1
    count_less = np.sum(perm_stats < original_stat)
    rank = count_less + 1
    
    # 4. Determine Pass/Fail (NIST Thresholds)
    # FAIL logic is different here compared to other tests?
    # No, NIST 800-90B applies the same two-tailed test to ALL statistics.
    # However, logically, compression failure implies T_original is LOW (rank small).
    # Rank <= 5 means original data compressed significantly better than random shuffles.
    # Rank >= 9995 means original data compressed significantly WORSE (unlikely for physical sources, 
    # but theoretically possible if shuffles accidentally create patterns).
    
    passed = True
    if rank <= 5 or rank >= (iterations - 5):
        passed = False
        
    return passed, rank, original_stat

# --- Usage Example ---
if __name__ == "__main__":
    # Case A: Random Data (Should Pass)
    # Random data is incompressible. T should be high.
    # Shuffles will also be random. T_orig should be near T_permuted mean.
    np.random.seed(42)
    random_data = np.random.randint(0, 256, 1000)
    
    print("Running Compression Test on Random Data...")
    passed, rank, stat = run_compression_test(random_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001 (Ideal range: 6-9994)")
    print(f"Statistic (Additions): {stat}\n")

    # Case B: Highly Compressible Data (Should Fail - Low Rank)
    # Repeating pattern: 0, 1, 0, 1...
    # Dictionary quickly learns "01", "0101", etc.
    # T will be very low (few additions needed).
    # Shuffling destroys the "01" pattern, making it harder to compress (Higher T).
    # Thus, Original T < Permuted T -> Rank 1.
    pattern_data = np.array([i % 2 for i in range(1000)])
    
    print("Running Compression Test on Pattern Data...")
    passed, rank, stat = run_compression_test(pattern_data)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Rank: {rank}/10001")
    print(f"Statistic (Additions): {stat} (Expect low rank)")