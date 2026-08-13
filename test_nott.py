import numpy as np
from scipy.stats import chi2

def _generate_templates(m=9):
    """Generates the 148 aperiodic templates for m=9."""
    templates = []
    for i in range(2**m):
        # Create candidate template
        bits = np.array([int(b) for b in format(i, f'0{m}b')], dtype=int)
        
        # Check Aperiodicity
        is_aperiodic = True
        for k in range(1, m):
            if np.array_equal(bits[:m-k], bits[k:]):
                is_aperiodic = False
                break
        
        if is_aperiodic:
            templates.append(bits)
            
    return templates

def non_overlapping_templates_test(bit_array: np.ndarray, M: int = 1032, alpha: float = 0.01) -> tuple:
    """
    Runs the test on ALL 148 aperiodic templates.
    Returns: (pass_count, total_count, average_p_value, best_p_value, worst_p_value)
    """
    n = len(bit_array)
    N = n // M
    if N < 8:
        # Not enough data for M=1032
        return 0, 148, 0.0, 0.0, 0.0

    # 1. Generate Templates
    templates = _generate_templates(m=9) # This generates 148 templates
    
    # 2. Setup Constants
    m = 9
    mu = (M - m + 1) / (2**m)
    sigma2 = M * ((1.0 / (2**m)) - ((2 * m - 1) / (2**(2 * m))))
    
    # 3. Optimize Data
    T_bits = bit_array[:N * M]
    blocks = T_bits.reshape(N, M)
    # Convert blocks to strings for fast counting
    block_strs = ["".join(map(str, block)) for block in blocks]
    
    pass_count = 0
    p_values = []
    
    # 4. Run Test Loop
    for tmpl in templates:
        tmpl_str = "".join(map(str, tmpl))
        W_counts = np.zeros(N)
        
        for i in range(N):
            W_counts[i] = block_strs[i].count(tmpl_str)
            
        chi_squared = np.sum((W_counts - mu)**2 / sigma2)
        p_val = chi2.sf(chi_squared, N)
        
        p_values.append(p_val)
        if p_val >= alpha:
            pass_count += 1
            
    # 5. Aggregate Results
    avg_p = np.mean(p_values)
    min_p = np.min(p_values)
    max_p = np.max(p_values)
    
    return pass_count, len(templates), avg_p, max_p, min_p