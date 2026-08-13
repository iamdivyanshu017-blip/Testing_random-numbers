import numpy as np
from scipy.stats import norm, chi2

def play_craps_game(rng_stream):
    """
    Simulates a single game of craps.
    Returns (win_bool, num_throws).
    """
    # Helper to roll two dice
    def roll():
        return int(next(rng_stream) * 6) + 1 + int(next(rng_stream) * 6) + 1

    throws = 1
    first_roll = roll()      

    # Instant Win
    if first_roll in [7, 11]:
        return True, throws
    # Instant Loss
    if first_roll in [2, 3, 12]:
        return False, throws

    # Point phase
    point = first_roll
    while True:
        throws += 1
        current_roll = roll()
        if current_roll == point:
            return True, throws
        if current_roll == 7:
            return False, throws

def craps_test(rng_func, num_games=200000):
    """
    Production-level Craps Test.
    """
    print(f"Executing Craps Test: Simulating {num_games} games...")
    
    # Pre-generate random numbers for performance
    # Average game is ~3.3 throws (6.6 random numbers), 
    # so we pull a large buffer.
    raw_buffer = rng_func(num_games * 10)
    rng_stream = iter(raw_buffer)
    
    wins = 0
    throws_dist = np.zeros(21) # Bins for 1, 2, ..., 20, >20 throws

    for _ in range(num_games):
        try:
            win, num_throws = play_craps_game(rng_stream)
            if win: wins += 1
            
            # Bin the throws (Cap at 21 for the >20 bin)
            bin_idx = min(num_throws, 21) - 1
            throws_dist[bin_idx] += 1
        except StopIteration:
            break
    
    # 1. Statistical Analysis of WINS
    # Theoretical P(win) = 244/495 ≈ 0.49292929
    p_win = 244 / 495
    expected_wins = num_games * p_win
    std_wins = np.sqrt(num_games * p_win * (1 - p_win))
    z_score = (wins - expected_wins) / std_wins
    p_val_wins = 1 - norm.cdf(z_score)

    # 2. Statistical Analysis of THROWS
    # Theoretical probabilities for throws 1, 2, ... 21+
    # These are defined by the Markov Chain of the Craps game
    theoretical_throws_p = np.array([
        1/3, 0.1388889, 0.1157407, 0.0950617, 0.0781204, 
        0.0641975, 0.0527552, 0.0433528, 0.0356263, 0.0292770,
        0.0240592, 0.0197711, 0.0162473, 0.0133516, 0.0109719,
        0.0090164, 0.0074095, 0.0060890, 0.0050038, 0.0041120, 
        0.0338450 # Bin for >20
    ])
    
    expected_throws = theoretical_throws_p * num_games
    # Chi-square test on throws distribution
    chi_stat = np.sum((throws_dist - expected_throws)**2 / expected_throws)
    p_val_throws = 1 - chi2.cdf(chi_stat, df=20)

    print("-" * 40)
    print(f"Wins: {wins} (Expected: {expected_wins:.2f})")
    print(f"P-Value (Wins):   {p_val_wins:.6f}")
    print(f"P-Value (Throws): {p_val_throws:.6f}")
    print("-" * 40)
    
    # --- INSERT DEBUG TABLE HERE ---
    print("Throw Distribution Analysis:")
    print(f"{'Throws':<8} | {'Observed':<10} | {'Expected':<10} | {'Diff':<10}")
    print("-" * 45)
    for i in range(21):
        label = f"{i+1}" if i < 20 else ">20"
        diff = throws_dist[i] - expected_throws[i]
        print(f"{label:<8} | {int(throws_dist[i]):<10} | {int(expected_throws[i]):<10} | {int(diff):<10}")
    print("-" * 40)

    if 0.0001 < p_val_wins < 0.9999 and 0.0001 < p_val_throws < 0.9999:
        print("RESULT: PASSED")
    else:
        print("RESULT: FAILED")
        
    return p_val_wins, p_val_throws

# Helper for standard RNG
def standard_rng(n):
    return np.random.random(n)

if __name__ == "__main__":
    craps_test(standard_rng)