import os
import sys
import math

# --- 1. PROCEDURE A: STATISTICAL TESTS (T0 - T5) ---

def test_t0_disjointness(bits):
    BLOCK_SIZE, REQUIRED_BLOCKS = 48, 65536
    REQUIRED_BITS = BLOCK_SIZE * REQUIRED_BLOCKS # 3,145,728 bits
    if len(bits) < REQUIRED_BITS:
        return False, {"error": f"Need {REQUIRED_BITS} bits"}
    seen = set()
    for i in range(0, REQUIRED_BITS, BLOCK_SIZE):
        val = int(bits[i : i + BLOCK_SIZE], 2)
        if val in seen: return False, {"duplicate": hex(val), "index": i // BLOCK_SIZE}
        seen.add(val)
    return True, {"count": len(seen)}

def test_t1_monobit(bits):
    n1 = bits[:20000].count('1')
    return 9654 < n1 < 10346, {"ones": n1}

def test_t2_poker(bits):
    counts = [0] * 16
    for i in range(0, 20000, 4):
        counts[int(bits[i:i+4], 2)] += 1
    y = (16/5000 * sum(f**2 for f in counts)) - 5000
    return 2.16 < y < 46.17, {"stat": round(y, 3)}

def test_t3_runs(bits):
    limits = {1: (2267, 2733), 2: (1079, 1421), 3: (502, 748), 4: (223, 402), 5: (90, 223), 6: (90, 223)}
    r0, r1 = {i:0 for i in range(1,7)}, {i:0 for i in range(1,7)}
    curr_bit, curr_len = bits[0], 0
    for bit in bits[:20000]:
        if bit == curr_bit: curr_len += 1
        else:
            d = r1 if curr_bit == '1' else r0
            d[min(curr_len, 6)] += 1
            curr_bit, curr_len = bit, 1
    return all(low < r0[i] < high and low < r1[i] < high for i, (low, high) in limits.items()), {"zeros": r0, "ones": r1}

def test_t4_long_run(bits):
    max_run, curr_bit, curr_len = 0, bits[0], 0
    for bit in bits[:20000]:
        if bit == curr_bit: curr_len += 1
        else:
            max_run, curr_bit, curr_len = max(max_run, curr_len), bit, 1
    return max(max_run, curr_len) < 34, {"max": max(max_run, curr_len)}

def test_t5_autocorr(bits):
    count = sum(1 for i in range(19999) if bits[i] == bits[i+1])
    return 9768 < count < 10232, {"d1_count": count}

# --- 2. PROCEDURE B: INTERNAL & ENTROPY TESTS (T6 - T8) ---

def test_t6_uniform(bits):
    n = 12500 # 100,000 bits
    counts = [0] * 256
    for i in range(0, n * 8, 8):
        counts[int(bits[i:i+8], 2)] += 1
    chi_sq = sum((f - (n/256))**2 / (n/256) for f in counts)
    return chi_sq < 311.56, {"chi_sq": round(chi_sq, 2)}

def test_t7_homogeneity(bits):
    def get_c(b):
        c = [0] * 256
        for i in range(0, 100000, 8): c[int(b[i:i+8], 2)] += 1
        return c
    c1, c2 = get_c(bits[:100000]), get_c(bits[100000:200000])
    t_stat = sum(((c1[i] - c2[i])**2 / (c1[i] + c2[i])) if (c1[i]+c2[i]) > 0 else 0 for i in range(256))
    return t_stat < 311.56, {"t_stat": round(t_stat, 2)}

def test_t8_entropy(bits):
    L, Q, K = 8, 2560, 25600 # 225,280 bits total
    last_seen, sum_log = {}, 0.0
    for i in range(Q):
        last_seen[int(bits[i*L : (i+1)*L], 2)] = i
    for i in range(Q, Q + K):
        block = int(bits[i*L : (i+1)*L], 2)
        if block in last_seen: sum_log += math.log2(i - last_seen[block])
        else: sum_log += math.log2(i + 1)
        last_seen[block] = i
    fn = sum_log / K
    return fn > 7.976, {"entropy_per_byte": round(fn, 5)}

# --- 3. INPUT & EXECUTION ---

def run_suite():
    print("\n" + "="*60 + "\n          AIS-31 COMPLETE SUITE (T0 - T8)\n" + "="*60)
    print("1. Read File | 2. Paste String")
    choice = input("Choice: ").strip()
    
    seq = ""
    if choice == '1':
        path = input("Path: ").strip().strip('"')
        if os.path.exists(path):
            with open(path, 'r') as f: seq = "".join(c for c in f.read() if c in '01')
    else:
        print("Paste binary:")
        seq = "".join(c for c in input().strip() if c in '01')

    L = len(seq)
    if L < 20000:
        print(f"❌ Not enough data ({L} bits). Minimum 20,000 required."); return

    results = {}
    # Map tests to bit requirements
    suite = [
        ("T1 Monobit", test_t1_monobit, 20000),
        ("T2 Poker", test_t2_poker, 20000),
        ("T3 Runs", test_t3_runs, 20000),
        ("T4 Long Run", test_t4_long_run, 20000),
        ("T5 Autocorr", test_t5_autocorr, 20000),
        ("T6 Uniform", test_t6_uniform, 100000),
        ("T7 Homogen", test_t7_homogeneity, 200000),
        ("T8 Entropy", test_t8_entropy, 225280),
        ("T0 Disjoint", test_t0_disjointness, 3145728)
    ]

    print(f"\n{'TEST':<12} | {'STATUS':<8} | {'DETAILS'}\n" + "-"*60)
    for name, func, req in suite:
        if L >= req:
            passed, info = func(seq)
            print(f"{name:<12} | {'✅ PASS' if passed else '❌ FAIL'} | {info}")
        else:
            print(f"{name:<12} | ⚠️ SKIP  | Needs {req} bits (Have {L})")
    print("="*60)

if __name__ == "__main__":
    run_suite()
    input("\nPress Enter to exit...")