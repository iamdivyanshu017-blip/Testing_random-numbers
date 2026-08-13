import numpy as np
from scipy import stats
from typing import Tuple, Union

def run_chi_square_independence(
    sample_data: Union[list, np.ndarray], 
    alphabet_size: int = 256,
    alpha: float = 0.001,
    verbose: bool = False
) -> Tuple[bool, float, float]:
    """
    Performs the Chi-Square Test for Independence (NIST SP 800-90B).
    
    Checks if adjacent samples are independent by analyzing the transition matrix.
    
    Args:
        sample_data: Input entropy source data.
        alphabet_size: Number of unique symbols (k). 
                       Use 2 for binary, 256 for bytes.
        alpha: Significance level (0.001 standard).
        verbose: Print stats.

    Returns:
        (passed, p_value, chi_square_stat)
    """
    data = np.array(sample_data, dtype=int)
    n = len(data)
    
    if n < 2:
        raise ValueError("Data must have at least 2 samples.")

    # 1. Construct Contingency Table (Observed Counts)
    # Rows: x_{i-1} (Previous), Cols: x_i (Current)
    # We count transitions (prev -> curr)
    
    # We use a 2D histogram or manual mapping.
    # For k=256, a 2D array is manageable (65k ints).
    observed = np.zeros((alphabet_size, alphabet_size), dtype=np.int64)
    
    # Fast vectorized accumulation using numpy
    # We treat (prev, curr) as a 1D index: prev * k + curr
    prev = data[:-1]
    curr = data[1:]
    
    flat_indices = prev * alphabet_size + curr
    
    # Count occurrences of each flat index
    counts = np.bincount(flat_indices, minlength=alphabet_size*alphabet_size)
    
    # Reshape back to k x k matrix
    observed = counts.reshape((alphabet_size, alphabet_size))
    
    # 2. Calculate Expected Counts under Independence Assumption
    # E_{r,c} = (RowTotal_r * ColTotal_c) / TotalObservations
    
    total_obs = n - 1
    row_totals = np.sum(observed, axis=1) # Sum of each row
    col_totals = np.sum(observed, axis=0) # Sum of each col
    
    # Check for empty rows/cols (can happen with small N or sparse data)
    # If a row or col is 0, E is 0. If O is also 0, contribution is 0.
    # We handle division by zero carefully.
    
    # Outer product to get matrix of (RowTotal * ColTotal)
    expected_matrix = np.outer(row_totals, col_totals) / total_obs
    
    # 3. Calculate Chi-Square Statistic
    # T = Sum( (O - E)^2 / E )
    
    # Mask to ignore cells where Expected is 0 (to avoid division by zero)
    # If Expected is 0, Observed MUST be 0 (since RowTotal or ColTotal was 0).
    # So (0-0)^2 / 0 is treated as 0 contribution.
    valid_mask = expected_matrix > 0
    
    # Calculate terms only for valid cells
    o_valid = observed[valid_mask]
    e_valid = expected_matrix[valid_mask]
    
    terms = (o_valid - e_valid)**2 / e_valid
    chi_sq_stat = np.sum(terms)
    
    # 4. Degrees of Freedom
    # df = (rows - 1)(cols - 1)
    # However, if some rows/cols are completely empty (never observed), 
    # the effective alphabet size is smaller.
    # NIST strictly usually assumes k is fixed, but standard stats suggests adjusting df.
    # We stick to theoretical df = (k-1)^2 for strict compliance, 
    # but be aware of sparsity issues.
    df = (alphabet_size - 1) ** 2
    
    # 5. P-Value
    p_value = stats.chi2.sf(chi_sq_stat, df)
    
    passed = p_value >= alpha
    
    if verbose:
        print(f"--- Chi Square Independence Test ---")
        print(f"Alphabet Size: {alphabet_size}")
        print(f"Degrees of Freedom: {df}")
        print(f"Statistic: {chi_sq_stat:.4f}")
        print(f"P-Value: {p_value:.6e}")
        print(f"Result: {'PASS (Independent)' if passed else 'FAIL (Dependent)'}")
        
        # Sparsity warning
        zero_e = np.sum(expected_matrix < 5)
        total_cells = alphabet_size * alphabet_size
        if zero_e > 0:
            print(f"Warning: {zero_e}/{total_cells} cells have Expected < 5.")
            print("Results may be unreliable for small N or large K.")

    return passed, p_value, chi_sq_stat

# --- Usage Example ---
if __name__ == "__main__":
    # Case A: Binary Data (k=2) - Random
    np.random.seed(42)
    binary_data = np.random.randint(0, 2, 1000)
    
    print("Running Independence Test on Binary Random Data...")
    run_chi_square_independence(binary_data, alphabet_size=2, verbose=True)
    
    # Case B: Binary Data - Alternating (Dependent)
    # 0, 1, 0, 1, 0, 1...
    # Previous value PERFECTLY predicts current value (0->1, 1->0).
    # Contingency table will look like:
    #       Curr 0   Curr 1
    # Prev 0   0       500
    # Prev 1  500       0
    # Expected would be 250 in each. Huge Chi-Square stat.
    alt_data = np.array([i % 2 for i in range(1000)])
    
    print("\nRunning Independence Test on Alternating Data...")
    run_chi_square_independence(alt_data, alphabet_size=2, verbose=True)

    # Case C: Byte Data (k=256) - Random
    # Note: Requires N >> k^2 for reliable p-values (N > 327,000 for full k=256)
    # With N=10,000, we have huge sparsity.
    # This is just a code demo.
    byte_data = np.random.randint(0, 256, 10000)
    print("\nRunning Independence Test on Byte Data (N=10k)...")
    run_chi_square_independence(byte_data, alphabet_size=256, verbose=True)