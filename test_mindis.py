import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import kstest, expon

def get_min_distance_sq(rng_func, n=8000, side=10000):
    """
    Places n points in a side x side square and returns 
    the square of the minimum distance found.
    """
    # Generate points (n, 2) scaled to the side length
    points = rng_func(n * 2).reshape(n, 2) * side
    
    # Use cKDTree (C-optimized KD-Tree) for fast neighbor lookup
    tree = cKDTree(points)
    
    # query(k=2) returns the distance to the point itself (0) 
    # and the nearest neighbor
    distances, _ = tree.query(points, k=2)
    
    # We want the minimum of the second column (the nearest neighbor distances)
    d_min = np.min(distances[:, 1])
    
    return d_min**2

def minimum_distance_test(rng_func, num_trials=100):
    """
    Production-level Minimum Distance Test.
    """
    print(f"Executing Minimum Distance Test: {num_trials} trials...")
    
    n = 8000
    side = 10000
    d_min_sq_list = []
    
    for i in range(num_trials):
        d_sq = get_min_distance_sq(rng_func, n, side)
        d_min_sq_list.append(d_sq)
        if (i+1) % 20 == 0:
            print(f"Completed {i+1}/{num_trials} trials...")

    # Theoretical Logic:
    # In a square of area A, with n points, the probability that 
    # the square of the minimum distance d^2 is greater than x is:
    # P(d^2 > x) = exp(- (n*(n-1)*pi*x) / (2 * side^2))
    # This is an exponential distribution with scale = (2 * side^2) / (n*(n-1)*pi)
    
    expected_scale = (2 * (side**2)) / (n * (n - 1) * np.pi)
    
    # Perform Kolmogorov-Smirnov test against the exponential distribution
    # This checks the entire distribution of results, not just the mean.
    res = kstest(d_min_sq_list, 'expon', args=(0, expected_scale))
    p_value = res.pvalue

    print("-" * 40)
    print(f"Expected Scale: {expected_scale:.6f}")
    print(f"Observed Mean:  {np.mean(d_min_sq_list):.6f}")
    print(f"KS Statistic:   {res.statistic:.4f}")
    print(f"P-Value:        {p_value:.6f}")

    if 0.0001 < p_value < 0.9999:
        print("RESULT: PASSED")
    else:
        print("RESULT: FAILED")
        
    return p_value

def standard_rng(n):
    return np.random.random(n)

if __name__ == "__main__":
    minimum_distance_test(standard_rng)