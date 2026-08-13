import math
import matplotlib.pyplot as plt
from collections import Counter

def calculate_mcv_estimate(sequence):
    n = len(sequence)
    counts = Counter(sequence)
    p_max = max(counts.values()) / n
    h_min = -math.log2(p_max)
    return h_min, p_max

def calculate_markov_estimate(sequence):
    n = len(sequence)
    transitions = {"00": 0, "01": 0, "10": 0, "11": 0}
    for i in range(n - 1):
        pair = sequence[i] + sequence[i+1]
        transitions[pair] += 1
    
    c0 = sequence[:-1].count("0")
    c1 = sequence[:-1].count("1")
    
    # Probabilities of transitions
    p_vals = [
        transitions["00"] / c0 if c0 > 0 else 0,
        transitions["01"] / c0 if c0 > 0 else 0,
        transitions["10"] / c1 if c1 > 0 else 0,
        transitions["11"] / c1 if c1 > 0 else 0
    ]
    p_max = max(p_vals)
    h_min = -math.log2(p_max) if p_max > 0 else 0
    return h_min, p_max

def plot_results(sequence):
    # Data for Histogram
    counts = Counter(sequence)
    bits = ['0', '1']
    frequencies = [counts.get('0', 0), counts.get('1', 0)]

    # 1. Bit Distribution Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(bits, frequencies, color=['#3498db', '#2ecc71'], alpha=0.8)
    plt.title('Bit Distribution')
    plt.ylabel('Frequency')

    # 2. Sequence Visual Trace (First 100 bits)
    plt.subplot(1, 2, 2)
    numeric_seq = [int(b) for b in sequence[:100]]
    plt.step(range(len(numeric_seq)), numeric_seq, where='post', color='#9b59b6')
    plt.title('Sequence Visual (First 100 Bits)')
    plt.ylim(-0.5, 1.5)
    plt.yticks([0, 1])
    
    plt.tight_layout()
    plt.savefig('entropy_analysis.png')
    print("\n[Visuals saved as 'entropy_analysis.png']")

def run_suite():
    user_input = input("Enter binary sequence: ").strip()
    if not set(user_input).issubset({'0', '1'}):
        print("Invalid input.")
        return

    h_mcv, p_mcv = calculate_mcv_estimate(user_input)
    h_markov, p_markov = calculate_markov_estimate(user_input)
    final_h_min = min(h_mcv, h_markov)

    print(f"\nMCV Entropy:    {h_mcv:.4f} bits")
    print(f"Markov Entropy: {h_markov:.4f} bits")
    print(f"Final H-Min:    {final_h_min:.4f} bits/bit")
    
    plot_results(user_input)

if __name__ == "__main__":
    run_suite()