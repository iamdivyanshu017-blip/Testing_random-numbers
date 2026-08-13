import numpy as np
from scipy import stats
from typing import Tuple, Union, Optional

class ChiSquareIndependence:
    """
    Implements NIST SP 800-90B Chi-Square Independence Test.
    Optimized for both Binary (k=2) and Non-Binary (k=256) data.
    """
    
    @staticmethod
    def _run_binary_fast(data: np.ndarray, alpha: float) -> Tuple[bool, float, float]:
        """
        Fast-path implementation for Binary Data (k=2).
        Uses a 1D optimization instead of 2D matrix operations.
        """
        n = len(data)
        
        # 1. Map pairs (x_i, x_{i+1}) to single integers 0..3
        # 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        # This allows us to use bincount once.
        pairs = 2 * data[:-1] + data[1:]
        counts = np.bincount(pairs, minlength=4)
        
        # Unpack counts into the 2x2 contingency table components
        # n00, n01, n10, n11
        obs = counts.reshape((2, 2))
        
        # 2. Compute Marginals
        row_totals = obs.sum(axis=1)
        col_totals = obs.sum(axis=0)
        total_pairs = n - 1
        
        # 3. Compute Chi-Square Statistic
        # T = Sum( (O - E)^2 / E )
        chi_sq_stat = 0.0
        
        # We manually iterate the 2x2 cells for speed/clarity and to handle /0 checks
        for r in range(2):
            for c in range(2):
                expected = (row_totals[r] * col_totals[c]) / total_pairs
                
                # Avoid division by zero if Expected is 0 (implies Observed is also 0)
                if expected > 0:
                    chi_sq_stat += ((obs[r, c] - expected) ** 2) / expected

        # 4. P-Value
        # Degrees of Freedom for k=2 is (2-1)*(2-1) = 1
        df = 1
        p_value = stats.chi2.sf(chi_sq_stat, df)
        
        passed = p_value >= alpha
        return passed, p_value, chi_sq_stat

    @staticmethod
    def _run_non_binary_general(data: np.ndarray, k: int, alpha: float) -> Tuple[bool, float, float]:
        """
        General implementation for Non-Binary Data (k > 2).
        Constructs a k x k transition matrix.
        """
        n = len(data)
        
        # 1. Construct k x k Contingency Table
        # Optimize using flat indexing: index = prev * k + curr
        prev = data[:-1]
        curr = data[1:]
        
        # Validate data range
        if np.max(data) >= k:
             raise ValueError(f"Data contains values >= {k}, but alphabet_size is {k}.")

        flat_indices = prev.astype(np.int64) * k + curr.astype(np.int64)
        counts = np.bincount(flat_indices, minlength=k*k)
        observed = counts.reshape((k, k))
        
        # 2. Compute Expected Frequencies
        row_totals = observed.sum(axis=1)
        col_totals = observed.sum(axis=0)
        total_pairs = n - 1
        
        # E matrix = (RowTotal * ColTotal) / N
        # Use outer product for fast calculation
        expected = np.outer(row_totals, col_totals) / total_pairs
        
        # 3. Compute Chi-Square Statistic
        # Valid mask: Only calculate where Expected > 0
        valid = expected > 0
        
        # Vectorized calculation
        terms = np.zeros_like(observed, dtype=np.float64)
        terms[valid] = ((observed[valid] - expected[valid]) ** 2) / expected[valid]
        
        chi_sq_stat = np.sum(terms)
        
        # 4. P-Value
        # df = (k-1)^2
        df = (k - 1) ** 2
        p_value = stats.chi2.sf(chi_sq_stat, df)
        
        passed = p_value >= alpha
        
        # Sparsity Warning for Production
        # If too many bins have E < 5, the test is statistically weak.
        # We don't fail the test, but we log/warn in a real system.
        # low_expected_count = np.sum((expected < 5) & (expected > 0))
        
        return passed, p_value, chi_sq_stat

    @classmethod
    def run_test(cls, 
                 sample_data: Union[list, np.ndarray], 
                 alphabet_size: Optional[int] = None, 
                 alpha: float = 0.001,
                 verbose: bool = False) -> Tuple[bool, float, float]:
        """
        Main entry point. Auto-detects data type if alphabet_size is None.
        
        Args:
            sample_data: The entropy source output.
            alphabet_size: 2 for binary, 256 for bytes. If None, inferred from max value.
            alpha: Significance level (default 0.001).
            verbose: Print results.
            
        Returns:
            (passed, p_value, statistic)
        """
        data = np.array(sample_data, dtype=int)
        
        # Auto-detect alphabet size if not provided
        if alphabet_size is None:
            max_val = np.max(data)
            if max_val <= 1:
                alphabet_size = 2
            elif max_val <= 255:
                alphabet_size = 256
            else:
                alphabet_size = max_val + 1
                
        if len(data) < 2:
            return False, 0.0, 0.0

        # Dispatch to appropriate implementation
        if alphabet_size == 2:
            passed, p_val, stat = cls._run_binary_fast(data, alpha)
            mode = "Binary (Fast Path)"
        else:
            passed, p_val, stat = cls._run_non_binary_general(data, alphabet_size, alpha)
            mode = f"Non-Binary (k={alphabet_size})"

        if verbose:
            print(f"--- Chi Square Independence Test ---")
            print(f"Mode: {mode}")
            print(f"Sample Size: {len(data)}")
            print(f"Statistic: {stat:.4f}")
            print(f"P-Value: {p_val:.6e}")
            print(f"Result: {'PASS' if passed else 'FAIL'}")
            
        return passed, p_val, stat

# --- Usage Example ---
if __name__ == "__main__":
    # 1. Binary Data Case
    # Alternating data (Highly Dependent) -> Should Fail
    print("Testing Binary Data (Alternating)...")
    binary_data = [0, 1] * 5000
    ChiSquareIndependence.run_test(binary_data, verbose=True)
    
    print("\n" + "-"*30 + "\n")
    
    # 2. Non-Binary Data Case
    # Random Bytes -> Should Pass
    print("Testing Non-Binary Data (Random Bytes)...")
    np.random.seed(42)
    # Note: N should be large for k=256 to avoid sparsity warnings (ideally > 300k)
    # We use N=100k here for demonstration speed.
    byte_data = np.random.randint(0, 256, 100000)
    ChiSquareIndependence.run_test(byte_data, alphabet_size=256, verbose=True)