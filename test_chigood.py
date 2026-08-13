import numpy as np
from scipy import stats
from typing import Tuple, Union

def run_chi_square_goodness_of_fit(
    sample_data: Union[list, np.ndarray], 
    alphabet_size: int = 256,
    alpha: float = 0.001,
    verbose: bool = False
) -> Tuple[bool, float, float]:
    """
    Performs the Chi-Square Goodness-of-Fit test for Uniformity (NIST SP 800-90B).
    
    Hypothesis:
        H0: The data is uniformly distributed.
        Ha: The data is not uniformly distributed.
        
    Args:
        sample_data: Input entropy source data.
        alphabet_size: The number of possible values (k). 
                       Default is 256 (for byte data). 
                       Use 2 for binary data.
        alpha: Significance level. NIST typically uses extremely low alphas (e.g., 0.001)
               for these tests to avoid false positives on large datasets.
        verbose: Print detailed stats.

    Returns:
        (passed, p_value, chi_square_stat)
        - passed (bool): True if p_value >= alpha (Fail to reject H0).
                         False if p_value < alpha (Reject H0 -> Non-Uniform).
        - p_value (float): The probability of observing this result if data were uniform.
        - chi_square_stat (float): The calculated test statistic.
    """
    data = np.array(sample_data)
    n = len(data)
    
    if n == 0:
        raise ValueError("Data cannot be empty.")

    # 1. Calculate Observed Frequencies (O_i)
    # np.bincount is fast, but we must ensure minlength=alphabet_size 
    # to account for bins that might have 0 counts.
    observed_counts = np.bincount(data.astype(int), minlength=alphabet_size)
    
    # If data contained values larger than alphabet_size, bincount extends automatically.
    # We must trim or validate.
    if len(observed_counts) > alphabet_size:
        raise ValueError(
            f"Data contains values >= {alphabet_size}, which exceeds the specified alphabet size."
        )

    # 2. Calculate Expected Frequencies (E_i) under H0 (Uniformity)
    # For uniform distribution, every bin expects N / k samples.
    expected_count = n / alphabet_size
    
    # Validation Warning: Chi-Square approximation is poor if expected_count < 5.
    if expected_count < 5 and verbose:
        print(f"Warning: Expected count ({expected_count:.2f}) is < 5. "
              "Chi-Square results may be unreliable for this sample size.")

    # 3. Calculate Chi-Square Statistic
    # Formula: Sum( (O - E)^2 / E )
    # Since E is constant, we can factor it out for slight speed: (1/E) * Sum((O-E)^2)
    
    diff_sq = (observed_counts - expected_count) ** 2
    chi_sq_stat = np.sum(diff_sq) / expected_count
    
    # 4. Calculate P-Value
    # Degrees of Freedom (df) = k - 1
    df = alphabet_size - 1
    
    # Use Survival Function (sf) = 1 - CDF
    # This provides better precision for very small p-values typical in crypto testing.
    p_value = stats.chi2.sf(chi_sq_stat, df)
    
    # 5. Determine Pass/Fail
    # If p_value < alpha, it is highly unlikely this data is uniform -> Fail.
    passed = p_value >= alpha
    
    if verbose:
        print(f"--- Chi Square Goodness of Fit ---")
        print(f"Sample Size (N): {n}")
        print(f"Bins (k): {alphabet_size}")
        print(f"Expected Count per Bin: {expected_count:.2f}")
        print(f"Chi-Square Statistic: {chi_sq_stat:.4f}")
        print(f"P-Value: {p_value:.6e}")
        print(f"Result: {'PASS (Uniform)' if passed else 'FAIL (Non-Uniform)'}")

    return passed, p_value, chi_sq_stat

# --- Usage Example ---
if __name__ == "__main__":
    # Case A: Random Uniform Data (Should Pass)
    # Generating truly uniform random integers
    np.random.seed(42)
    uniform_data = np.random.randint(0, 256, 10000)
    
    print("Running Chi-Square on Uniform Data...")
    run_chi_square_goodness_of_fit(uniform_data, alphabet_size=256, verbose=True)
    print("\n" + "="*30 + "\n")

    # Case B: Biased Data (Should Fail)
    # Generating data roughly Normal distribution centered at 128
    # This is "Random" but not "Uniform".
    normal_data = np.random.normal(128, 30, 10000).astype(int)
    # Clip to byte range
    normal_data = np.clip(normal_data, 0, 255)
    
    print("Running Chi-Square on Normal/Biased Data...")
    run_chi_square_goodness_of_fit(normal_data, alphabet_size=256, verbose=True)