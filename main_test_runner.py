import numpy as np
import time
from typing import List, Tuple
from utils import convert_input
from test_monobit import frequency_monobit_test
from test_runs import runs_test
from test_block_freq import block_frequency_test
from test_longest_run import longest_run_test
from test_spectral import spectral_test
from test_matrix_rank import matrix_rank_test
from test_approximate_entropy import approximate_entropy_test
from test_serial import serial_test
from test_cusum import cumulative_sums_test
from test_rev import random_excursions_variant_test
from test_nott import non_overlapping_templates_test
from test_ott import overlapping_templates_test
from test_universal import universal_test
from test_ret import random_excursions_test

ALPHA = 0.01

import random


def generate_random_bits(n_bits: int) -> str:
    """Generate exactly n_bits pseudorandom bits."""
    return ''.join(str(random.getrandbits(1)) for _ in range(n_bits))


def run_full_nist_suite(binary_data: str, alpha: float) -> List[Tuple]:
    results = []

    try:
        bit_array = convert_input(binary_data)
        n = len(bit_array)
        print(f"Starting test suite on {n} bits...")
    except ValueError as e:
        results.append(("Data Validation", 0.0, f"FAIL: {e}"))
        return results

    M_BLOCK = 128
    M_SERIAL = 10
    M_NOTT = 1032

    def safe(name, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            results.append((name, 0.0, f"EXECUTION ERROR: {e}"))
            return None

    # 1. Monobit Test
    r = safe(
        "1. Monobit Test",
        frequency_monobit_test,
        bit_array,
        alpha
    )
    if r:
        results.append(("1. Monobit Test", r[0], r[1]))

    # 2. Runs Test
    r = safe(
        "2. Runs Test",
        runs_test,
        bit_array,
        alpha
            )
    if r:
        results.append(("2. Runs Test", r[0], r[1]))

    # 3. Block Frequency Test
    r = safe(
        "3. Block Frequency Test",
        block_frequency_test,
        bit_array,
        M_BLOCK,
        alpha
    )
    if r:
        results.append(("3. Block Frequency Test", r[0], r[1]))

    # 4. Longest Run Test
    r = safe(
        "4. Longest Run Test",
        longest_run_test,
        bit_array,
        alpha
    )
    if r:
        results.append(("4. Longest Run Test", r[0], r[1]))

    # 5. Spectral Test
    r = safe(
        "5. Spectral Test",
        spectral_test,
        bit_array,
        alpha
    )
    if r:
        results.append(("5. Spectral Test", r[0], r[1]))

    # 6. Matrix Rank Test
    r = safe(
        "6. Matrix Rank Test",
        matrix_rank_test,
        bit_array,
        alpha
    )
    if r:
        results.append(("6. Matrix Rank Test", r[0], r[1]))

    # 7. Approximate Entropy Test
    # alpha must be passed by keyword because m is the second parameter.
    r = safe(
        "7. Approximate Entropy Test",
        approximate_entropy_test,
        bit_array,
        alpha=alpha
    )
    if r:
        results.append(("7. Approximate Entropy Test", r[0], r[1]))

    # 8. Serial Test
    r = safe(
        "8. Serial Test",
        serial_test,
        bit_array,
        M_SERIAL,
        alpha
    )
    if r:
        results.append(("8. Serial Test (Part I)", r[0], r[1]))
        results.append(("8. Serial Test (Part II)", r[2], r[3]))

    # 9. Cumulative Sums Test
    r = safe(
        "9. Cumulative Sums Test",
        cumulative_sums_test,
        bit_array,
        alpha
    )
    if r:
        results.append(("9. Cumulative Sums Test (Forward)", r[0], r[1]))
        results.append(("9. Cumulative Sums Test (Backward)", r[2], r[3]))

    # 10. Random Excursions Variant
    # Returns a list of (state, p-value, verdict) tuples.
    r = safe(
        "10. Random Excursions Variant Test",
        random_excursions_variant_test,
        bit_array,
        alpha
    )

    if r is not None:
        if r and all(
            isinstance(v, str) and v.startswith("SKIP")
            for _, _, v in r
        ):
            results.append(
                (
                    "10. Random Excursions Variant Test",
                    0.0,
                    "SKIP (J < 500 cycles)"
                )
            )
        elif not r:
            results.append(
                (
                    "10. Random Excursions Variant Test",
                    0.0,
                    "SKIP (J < 500 cycles)"
                )
            )
        else:
            for state, p, v in r:
                results.append(
                    (
                        f"10. Random Excursions Variant (State {state})",
                        p,
                        v
                    )
                )

    # 11. Non-Overlapping Templates
    r = safe(
        "11. Non-Overlapping Templates Test",
        non_overlapping_templates_test,
        bit_array,
        M_NOTT,
        alpha
    )
    if r:
        pass_count, total, avg_p, max_p, min_p = r
        results.append(
            (
                f"11. Non-Overlapping Templates "
                f"({pass_count}/{total} templates passed)",
                avg_p,
                pass_count >= total * 0.96
            )
        )

    # 12. Overlapping Templates
    r = safe(
        "12. Overlapping Templates Test",
        overlapping_templates_test,
        bit_array,
        alpha
    )
    if r:
        results.append(
            ("12. Overlapping Templates Test", r[0], r[1])
        )

    # 13. Universal Test
    r = safe(
        "13. Universal Test",
        universal_test,
        bit_array,
        alpha
    )
    if r:
        results.append(
            ("13. Universal Test", r[0], r[1])
        )

    # 14. Random Excursions
    # Returns a list of (state, p-value, verdict) tuples.
    r = safe(
        "14. Random Excursions Test",
        random_excursions_test,
        bit_array,
        alpha
    )

    if r is not None:
        if r and all(
            isinstance(v, str) and v.startswith("SKIP")
            for _, _, v in r
        ):
            results.append(
                (
                    "14. Random Excursions Test",
                    0.0,
                    "SKIP (J < 500 cycles)"
                )
            )
        elif not r:
            results.append(
                (
                    "14. Random Excursions Test",
                    0.0,
                    "SKIP (J < 500 cycles)"
                )
            )
        else:
            for state, p, v in r:
                results.append(
                    (
                        f"14. Random Excursions (State {state})",
                        p,
                        v
                    )
                )

    return results
if __name__ == '__main__':
    TEST_BITS = 10000000
    ALPHA = 0.01
    TEST_DATA_EXAMPLE = generate_random_bits(TEST_BITS)
    final_results = run_full_nist_suite(TEST_DATA_EXAMPLE, ALPHA)

    print("\n" + "=" * 80)
    print("        FINAL NIST SP 800-22 STATISTICAL TEST SUITE REPORT")
    print("=" * 80)
    print(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sequence Length (n): {len(TEST_DATA_EXAMPLE)} bits")
    print(f"Significance Level (Alpha): {ALPHA}")
    print("\n" + "-" * 80)

    header = ["#", "Test Name", "P-Value", "Verdict"]
    print(f"{header[0]:<3}{header[1]:<38}{header[2]:<15}{header[3]:<10}")
    print("-" * 66)

    i = 1
    for name, p_value, verdict in final_results:
        if isinstance(verdict, str):
            verdict_str = verdict
        else:
            verdict_str = "PASS" if verdict else "FAIL"
        p_str = f"{p_value:.6f}" if isinstance(p_value, float) else str(p_value)
        print(f"{i:<3}{name:<38}{p_str:<15}{verdict_str:<10}")
        i += 1

    print("-" * 66)

    countable = [v for _, _, v in final_results if isinstance(v, bool)]
    overall_pass = all(countable) if countable else False

    summary = "OVERALL RANDOMNESS VERDICT: "
    overall_pass_status = "PASS" if overall_pass else "FAIL (One or more tests failed)"
    print(summary + overall_pass_status)
    if overall_pass:
        print("\nCONCLUSION: The binary sequence IS statistically RANDOM.")
    else:
        print("\nCONCLUSION: The binary sequence IS NOT statistically random.")
    print("=" * 80)
