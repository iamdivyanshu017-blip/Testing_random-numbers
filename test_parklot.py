import numpy as np
from scipy.spatial import KDTree
from scipy.stats import norm

def run_single_parking_trial(rng_func, side=100.0, radius=1.0, attempts=12000):
    """
    Simulates one parking lot trial.
    Returns the number of successfully parked circles.
    """
    # Generate all attempts at once for speed
    # We need 2 random floats per attempt for (x, y)
    coords = rng_func(attempts * 2).reshape(attempts, 2) * side
    
    parked_cars = []
    
    for i in range(attempts):
        new_car = coords[i]
        
        if not parked_cars:
            parked_cars.append(new_car)
            continue
            
        # Optimization: Use a KD-Tree to find the nearest parked car
        # A car overlaps if the distance is less than 2*radius (diameter)
        tree = KDTree(parked_cars)
        dist, _ = tree.query(new_car)
        
        if dist >= 2 * radius:
            parked_cars.append(new_car)
            
    return len(parked_cars)

def parking_lot_test(rng_func, num_trials=100):
    """
    Production-level Parking Lot Test.
    
    Args:
        rng_func: Function returning random floats in [0, 1).
        num_trials: Number of independent parking simulations (Diehard uses 100).
    """
    print(f"Executing Parking Lot Test: {num_trials} trials...")
    
    success_counts = []
    for i in range(num_trials):
        count = run_single_parking_trial(rng_func)
        success_counts.append(count)
        if (i+1) % 10 == 0:
            print(f"Completed {i+1}/{num_trials} trials...")

    # Statistical Constants
    # For a 100x100 lot, 12000 attempts, and radius 1.0:
    # The mean number of successes is approx 3523 with stddev 21.9
    # (Constants based on Marsaglia's empirical and theoretical findings)
    theoretical_mean = 3523
    theoretical_stddev = 21.9
    
    obs_mean = np.mean(success_counts)
    z_score = (obs_mean - theoretical_mean) / (theoretical_stddev / np.sqrt(num_trials))
    p_value = 1 - norm.cdf(z_score)

    print("-" * 40)
    print(f"Average Parked Cars: {obs_mean:.2f}")
    print(f"Theoretical Mean:   {theoretical_mean}")
    print(f"Z-Score:            {z_score:.4f}")
    print(f"P-Value:            {p_value:.6f}")

    if 0.0001 < p_value < 0.9999:
        print("RESULT: PASSED")
    else:
        print("RESULT: FAILED")
        
    return p_value

# Helper for standard random floats
def standard_float_rng(n):
    return np.random.random(n)

if __name__ == "__main__":
    parking_lot_test(standard_float_rng)