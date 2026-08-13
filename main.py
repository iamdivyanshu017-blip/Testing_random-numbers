import sys
import os
import numpy as np

try:
    import test_excur
    import test_noruns
    import test_lenruns
    import test_noincdec
    import test_norunmedian
    import test_lenrunmedian
    import test_avgcol
    import test_maxcol
    import test_period
    import test_covariance
    import test_compress
    import test_chigood
    import test_indibinnonbin
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import test modules. Missing file: {e.name}")
    print("Ensure all test_*.py files are in the same folder.")
    sys.exit(1)
    
def pack_bits_to_bytes(bit_array):
    """
    Groups bits into non-overlapping 8-bit chunks and converts each 
    to a byte value (0-255). Standard NIST SP800-90B approach for 
    running symbol-based tests on binary (bit-level) data.
    """
    n_bytes = len(bit_array) // 8
    trimmed = bit_array[:n_bytes * 8]  # discard leftover bits that don't fill a byte
    reshaped = trimmed.reshape(n_bytes, 8)
    byte_values = np.zeros(n_bytes, dtype=int)
    for i in range(8):
        byte_values = byte_values * 2 + reshaped[:, i]
    return byte_values

def load_data(input_source):
    """
    Parses input into a numpy array of integers.
    Supports:
    1. File Path (Binary or Text file)
    2. Raw Binary String (e.g., "1101001")
    """
    data = []
    
    # Check if input is a file path that exists
    if os.path.isfile(input_source):
     print(f"Loading data from file: {input_source}")
     try:
        # First, try reading as a text file of 0s/1s
        with open(input_source, 'r') as f:
            text = f.read().strip()
        if all(c in '01' for c in text) and len(text) > 1:
            print("Detected binary-text file (0s and 1s).")
            data = [int(bit) for bit in text]
        else:
            # Fall back to raw binary byte reading
            with open(input_source, 'rb') as f:
                raw_bytes = f.read()
                if len(raw_bytes) < 50:
                    print(f"Raw bytes: {raw_bytes}")
                data = list(raw_bytes)
     except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    # Check if input is a binary string (e.g., "010101")
    elif all(c in '01' for c in input_source) and len(input_source) > 1:
        print("Detected raw binary string input.")
        data = [int(bit) for bit in input_source]
        
    else:
        print("Error: Input must be a valid file path or a binary string (0s and 1s).")
        return None

    return np.array(data, dtype=int)

def run_all_tests(data, verbose=False):
    """
    Runs the full NIST IID suite on the provided data.
    """
    results = {}
    overall_pass = True
    
    n = len(data)
    print(f"\n--- Starting IID Validation Suite ---")
    print(f"Sample Size: {n}")
    print(f"Alphabet Size: {len(np.unique(data))} unique symbols found.")
    print("-" * 50)

    # Helper wrapper to catch errors in individual tests preventing the whole suite from crashing
    def run_safe(test_name, func, *args):
        try:
            if verbose: print(f"Running {test_name}...")
            return func(*args)
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            return (False, -1, -1) # Fail safe

    # 1. Excursion Test
    passed, rank, stat = run_safe("Excursion Test", test_excur.run_iid_excursion_test, data)
    results['Excursion'] = passed
    
    # 2. Number of Directional Runs
    passed, rank, stat = run_safe("Num Directional Runs", test_noruns.run_directional_runs_test, data)
    results['Num Directional Runs'] = passed

    # 3. Length of Directional Runs
    passed, rank, stat = run_safe("Len Directional Runs", test_lenruns.run_len_directional_runs_test, data)
    results['Len Directional Runs'] = passed

    # 4. Number of Increases/Decreases
    passed, rank, stat = run_safe("Num Inc/Dec", test_noincdec.run_increases_decreases_test, data)
    results['Num Inc/Dec'] = passed

    # 5. Number of Runs (Median)
    passed, rank, stat = run_safe("Num Runs (Median)", test_norunmedian.run_num_runs_median_test, data)
    results['Num Runs (Median)'] = passed

    # 6. Length of Runs (Median)
    passed, rank, stat = run_safe("Len Runs (Median)", test_lenrunmedian.run_len_runs_median_test, data)
    results['Len Runs (Median)'] = passed

    # 7. Average Collision
    passed, rank, stat = run_safe("Average Collision", test_avgcol.run_avg_collision_test, data)
    results['Average Collision'] = passed

    # 8. Maximum Collision
    passed, rank, stat = run_safe("Max Collision", test_maxcol.run_max_collision_test, data)
    results['Max Collision'] = passed

    # 9. Periodicity (Returns multiple ranks)
    try:
        if verbose: print("Running Periodicity Test...")
        p_pass, p_ranks, p_stats = test_period.run_periodicity_test(data)
        results['Periodicity'] = p_pass
    except Exception as e:
        print(f"ERROR in Periodicity: {e}")
        results['Periodicity'] = False

    # 10. Covariance (Returns multiple ranks)
    try:
        if verbose: print("Running Covariance Test...")
        c_pass, c_ranks, c_stats = test_covariance.run_covariance_test(data)
        results['Covariance'] = c_pass
    except Exception as e:
        print(f"ERROR in Covariance: {e}")
        results['Covariance'] = False

    # 11. Compression
    passed, rank, stat = run_safe("Compression", test_compress.run_compression_test, data)
    results['Compression'] = passed

    # --- Chi-Square Tests ---
    
    # 12. Goodness of Fit (Uniformity)
    # Auto-detect alphabet size (2 for binary, 256 for bytes)
    k = 2 if np.max(data) <= 1 else 256
    try:
        if verbose: print("Running Chi-Square Goodness of Fit...")
        passed, p_val, stat = test_chigood.run_chi_square_goodness_of_fit(data, alphabet_size=k)
        results['Chi-Square Goodness of Fit'] = passed
    except Exception as e:
        print(f"ERROR in Chi-Square Goodness: {e}")
        results['Chi-Square Goodness of Fit'] = False

    # 13. Independence (Binary/Non-Binary)
    try:
        if verbose: print("Running Chi-Square Independence...")
        # Using the unified class we built
        passed, p_val, stat = test_indibinnonbin.ChiSquareIndependence.run_test(data, alphabet_size=k)
        results['Chi-Square Independence'] = passed
    except Exception as e:
        print(f"ERROR in Chi-Square Independence: {e}")
        results['Chi-Square Independence'] = False

    # --- Final Report ---
    print("\n" + "="*50)
    print(f"{'TEST NAME':<30} | {'RESULT':<10}")
    print("-" * 50)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:<30} | {status:<10}")
        if not passed:
            overall_pass = False
            
    print("-" * 50)
    if overall_pass:
        print("FINAL VERDICT: IID ASSUMPTION VALIDATED (Use IID Track)")
    else:
        print("FINAL VERDICT: IID ASSUMPTION REJECTED (Use Non-IID Track)")
    print("="*50 + "\n")

if __name__ == "__main__":
    print("NIST SP 800-90B IID Test Suite Runner")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nEnter binary string or file path: ").strip()
        
        if user_input.lower() == 'exit':
            break
            
        if not user_input:
            continue
            
        data = load_data(user_input)
        
        if data is not None and len(data) > 0:
            if len(data) < 1000:
                print("WARNING: Sample size < 1000. Results may be statistically unreliable.")

            if len(np.unique(data)) <= 2:
                print(f"Binary data detected. Packing {len(data)} bits into bytes...")
                data = pack_bits_to_bytes(data)
                print(f"Packed into {len(data)} byte-values (0-255).\n")

            run_all_tests(data, verbose=True)
