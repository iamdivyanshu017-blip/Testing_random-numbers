from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import traceback

# --- DYNAMIC IMPORTS ---
def safe_import(module_name, func_name):
    try:
        mod = __import__(module_name, fromlist=[func_name])
        return getattr(mod, func_name)
    except ImportError:
        print(f"WARNING: Could not import {func_name} from {module_name}.")
        return None
    except AttributeError:
        print(f"WARNING: {func_name} not found in {module_name}.")
        return None

# Import all tests
frequency_monobit_test = safe_import("test_monobit", "frequency_monobit_test")
block_frequency_test = safe_import("test_block_freq", "block_frequency_test")
runs_test = safe_import("test_runs", "runs_test")
longest_run_test = safe_import("test_longest_run", "longest_run_test")
matrix_rank_test = safe_import("test_matrix_rank", "matrix_rank_test")
spectral_test = safe_import("test_spectral", "spectral_test")
serial_test = safe_import("test_serial", "serial_test")
approximate_entropy_test = safe_import("test_approximate_entropy", "approximate_entropy_test")
cumulative_sums_test = safe_import("test_cusum", "cumulative_sums_test")
universal_test = safe_import("test_universal", "universal_test")
overlapping_templates_test = safe_import("test_ott", "overlapping_templates_test")
non_overlapping_templates_test = safe_import("test_nott", "non_overlapping_templates_test")
random_excursions_variant_test = safe_import("test_rev", "random_excursions_variant_test")
random_excursions_test = safe_import("test_ret", "random_excursions_test")
linear_complexity_test = safe_import("test_linear_complexity", "linear_complexity_test")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TestRequest(BaseModel):
    binary_data: str
    alpha: float = 0.01

def convert_to_bit_array(binary_string: str):
    clean = ''.join(c for c in binary_string if c in '01')
    return np.array(list(clean), dtype=int)

@app.post("/test_randomness")
async def run_tests(request: TestRequest):
    try:
        bit_array = convert_to_bit_array(request.binary_data)
        n = len(bit_array)
        if n < 100:
            raise HTTPException(status_code=400, detail=f"Sequence too short ({n} bits).")
        
        alpha = request.alpha
        results = []

        def add_res(name, p, ver, extra=""):
            try:
                # Force types to native Python for JSON serialization
                clean_p = round(float(p), 6)
            except (ValueError, TypeError):
                clean_p = str(p)
            
            try:
                clean_ver = bool(ver) if not isinstance(ver, str) else ver
            except:
                clean_ver = False

            results.append({
                "test_name": str(name),
                "p_value": clean_p,
                "verdict": clean_ver,
                "error": str(extra)
            })

        # --- TESTS 1-13 (Standard Calls) ---
        
        if frequency_monobit_test:
            try:
                p, v = frequency_monobit_test(bit_array, alpha)
                add_res("Frequency (Monobit)", p, v)
            except Exception as e: add_res("Frequency", 0, False, str(e))

        if block_frequency_test:
            try:
                p, v = block_frequency_test(bit_array, M=128, alpha=alpha)
                add_res("Block Frequency", p, v)
            except Exception as e: add_res("Block Frequency", 0, False, str(e))

        if runs_test:
            try:
                p, v = runs_test(bit_array, alpha)
                add_res("Runs", p, v)
            except Exception as e: add_res("Runs", 0, False, str(e))

        if longest_run_test:
            try:
                p, v = longest_run_test(bit_array, alpha=alpha)
                add_res("Longest Run", p, v)
            except Exception as e: add_res("Longest Run", 0, False, str(e))

        if matrix_rank_test:
            try:
                p, v = matrix_rank_test(bit_array, alpha=alpha)
                add_res("Binary Matrix Rank", p, v)
            except Exception as e: add_res("Matrix Rank", 0, False, str(e))

        if spectral_test:
            try:
                p, v = spectral_test(bit_array, alpha)
                add_res("Spectral (DFT)", p, v)
            except Exception as e: add_res("Spectral", 0, False, str(e))

        if serial_test:
            try:
                p1, v1, p2, v2 = serial_test(bit_array, m=16, alpha=alpha)
                add_res("Serial (P1)", p1, v1)
                add_res("Serial (P2)", p2, v2)
            except Exception as e: add_res("Serial", 0, False, str(e))

        if approximate_entropy_test:
            try:
                p, v = approximate_entropy_test(bit_array, m=10, alpha=alpha)
                add_res("Approximate Entropy", p, v)
            except Exception as e: add_res("Approximate Entropy", 0, False, str(e))

        if cumulative_sums_test:
            try:
                pf, vf, pb, vb = cumulative_sums_test(bit_array, alpha)
                add_res("Cusum (Forward)", pf, vf)
                add_res("Cusum (Backward)", pb, vb)
            except Exception as e: add_res("Cusum", 0, False, str(e))

        if universal_test:
            try:
                res = universal_test(bit_array, alpha=alpha)
                add_res("Maurer's Universal", res[0], res[1])
            except Exception as e: add_res("Universal", 0, False, str(e))

        if overlapping_templates_test:
            try:
                p, v = overlapping_templates_test(bit_array, alpha=alpha)
                add_res("Overlapping Template", p, v)
            except Exception as e: add_res("Overlapping Template", 0, False, str(e))

        if random_excursions_variant_test:
            try:
                rev_res = random_excursions_variant_test(bit_array, alpha=alpha)
                if not rev_res:
                    add_res("REV", "N/A", True, "Insufficient Cycles (J<500)")
                else:
                    # Report State +1 as representative
                    state_res = next((r for r in rev_res if r[0] == 1), None)
                    if state_res:
                        add_res("REV (State +1)", state_res[1], state_res[2])
                    else:
                        add_res("REV", "N/A", True)
            except Exception as e: add_res("REV", 0, False, str(e))

        if linear_complexity_test:
            try:
                p, v = linear_complexity_test(bit_array, M=500, alpha=alpha)
                add_res("Linear Complexity", p, v)
            except Exception as e: add_res("Linear Complexity", 0, False, str(e))

        # --- UPDATED TESTS (Aggregated Results) ---

        # 12. Non-Overlapping Template (Run 148 Tests)
        if non_overlapping_templates_test:
            try:
                # Returns: (pass_count, total_count, avg_p, max_p, min_p)
                passed, total, avg_p, max_p, min_p = non_overlapping_templates_test(bit_array, M=1032, alpha=alpha)
                
                # We report the "Worst P-value" (Minimum) to be conservative
                # Verdict is PASS only if Pass Rate > 96% (NIST standard allowance)
                pass_rate = passed / total
                final_verdict = pass_rate >= 0.96
                
                add_res(f"Non-Overlapping (Aggregate)", min_p, final_verdict, f"Passed {passed}/{total} templates")
            except Exception as e: add_res("Non-Overlapping", 0, False, str(e))

        # 15. Random Excursions (Run 8 States)
        if random_excursions_test:
            try:
                ret_res = random_excursions_test(bit_array, alpha=alpha)
                
                if isinstance(ret_res, tuple): # Insufficient Cycles
                    add_res("Random Excursions", "N/A", True, f"Insufficient Cycles (J={ret_res[2]})")
                elif isinstance(ret_res, list):
                    # Add row for EVERY state
                    for item in ret_res:
                        state, p_val, ver = item
                        add_res(f"Random Excursions (State {state})", p_val, ver)
                else:
                    add_res("Random Excursions", "N/A", False, "Unknown Return Type")
            except Exception as e: add_res("Random Excursions", 0, False, str(e))

        return results

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))