from __future__ import annotations
from typing import Tuple
import numpy as np


def simulate_grid(
    pace_index: np.ndarray,
    dnf_prob: np.ndarray,
    draws: int = 5000,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo simulation: for each draw, add noise to pace, apply DNFs, and derive finishing order.

    Args:
        pace_index: Lower values = faster driver
        dnf_prob: Probability of DNF for each driver (0 to 1)
        draws: Number of simulation iterations
        random_seed: For reproducibility

    Returns:
        - prob_matrix: [n_drivers, n_drivers] probability of finishing at each position (1..N)
        - mean_pos: expected finishing position for each driver
        - pairwise_matrix: [n_drivers, n_drivers] where entry (i,j) = P(i finishes ahead of j)
    """
    rng = np.random.RandomState(random_seed)
    n = len(pace_index)
    
    if n == 0:
        return np.array([]).reshape(0, 0), np.array([]), np.array([]).reshape(0, 0)
    
    counts = np.zeros((n, n), dtype=float)
    pairwise = np.zeros((n, n), dtype=float)

    # Calculate noise scale based on pace spread
    # We want enough noise for uncertainty but not so much that it overwhelms skill differences
    pace_std = np.std(pace_index)
    pace_range = np.ptp(pace_index)  # max - min
    
    # Noise should be proportional to the pace spread, but with a minimum
    # This allows ~10-20% chance of adjacent drivers swapping on pure noise
    if pace_range > 0.01:
        noise_scale = pace_range * 0.15
    else:
        noise_scale = 0.1
    
    # Ensure minimum noise for some randomness
    noise_scale = max(noise_scale, 0.05)
    
    # DNF penalty - move driver to back of grid
    max_penalty = abs(np.max(pace_index)) + abs(np.min(pace_index)) + 20.0

    for _ in range(draws):
        # Add performance noise
        noisy = pace_index + rng.normal(0.0, noise_scale, size=n)
        
        # Apply DNFs (driver gets large penalty, effectively placing them last)
        dnf_draw = rng.binomial(1, dnf_prob, size=n)
        sim = noisy + dnf_draw * max_penalty

        # Determine finishing order (lower sim value = better position)
        order = np.argsort(sim)
        for pos, idx in enumerate(order):
            counts[idx, pos] += 1.0

        # Calculate pairwise comparisons
        rank = np.empty(n, dtype=int)
        rank[order] = np.arange(n)
        r_i = rank.reshape(-1, 1)
        r_j = rank.reshape(1, -1)
        ahead = (r_i < r_j).astype(float)
        pairwise += ahead

    # Convert counts to probabilities
    prob_matrix = counts / draws
    
    # Calculate mean position (1-indexed)
    positions = np.arange(1, n + 1)
    mean_pos = (prob_matrix * positions).sum(axis=1)
    
    # Pairwise probability matrix
    pairwise_prob = pairwise / draws
    np.fill_diagonal(pairwise_prob, 0.5)
    
    return prob_matrix, mean_pos, pairwise_prob
