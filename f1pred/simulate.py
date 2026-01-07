"""Monte Carlo simulation for F1 race outcomes.

This module simulates race finishes by adding stochastic noise to pace
predictions and applying DNF probabilities.
"""
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

__all__ = ["simulate_grid"]


def simulate_grid(
    pace_index: np.ndarray,
    dnf_prob: np.ndarray,
    draws: int = 5000,
    random_seed: Optional[int] = 42,
    noise_factor: float = 0.15,
    min_noise: float = 0.05,
    max_penalty_base: float = 20.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo simulation: for each draw, add noise to pace, apply DNFs, and derive finishing order.

    Optimized implementation using vectorized numpy operations.

    Args:
        pace_index: Lower values = faster driver
        dnf_prob: Probability of DNF for each driver (0 to 1)
        draws: Number of simulation iterations
        random_seed: For reproducibility. If None, uses system entropy.
        noise_factor: Multiplier for pace spread to determine noise scale
        min_noise: Minimum noise standard deviation
        max_penalty_base: Base penalty added to max pace for DNF drivers
    """
    rng = np.random.RandomState(random_seed)
    n = len(pace_index)
    
    if n == 0:
        return np.array([]).reshape(0, 0), np.array([]), np.array([]).reshape(0, 0)
    
    # Calculate noise scale based on pace spread
    pace_range = float(np.ptp(pace_index))
    
    if pace_range > 0.01:
        noise_scale = pace_range * noise_factor
    else:
        noise_scale = 0.1
    noise_scale = max(noise_scale, min_noise)
    
    max_penalty = abs(np.max(pace_index)) + abs(np.min(pace_index)) + max_penalty_base

    # Generate all draws at once (vectorized)
    # Shape: (draws, n)
    noisy = pace_index + rng.normal(0.0, noise_scale, size=(draws, n))
    dnf_draw = rng.binomial(1, dnf_prob, size=(draws, n))
    sim = noisy + dnf_draw * max_penalty

    # Get ranks for each draw
    # argsort(sim, axis=1) gives indices of drivers sorted by sim (finishing order)
    # argsort again gives the rank of each driver (0-based)
    ranks = np.argsort(np.argsort(sim, axis=1), axis=1)

    # Calculate counts (probabilities of finishing position)
    counts = np.zeros((n, n), dtype=float)

    # For each driver (column in ranks), count occurrences of each rank
    for i in range(n):
        # ranks[:, i] contains the ranks obtained by driver i across all draws
        counts[i, :] = np.bincount(ranks[:, i], minlength=n)

    # Calculate pairwise comparisons
    # ranks shape (draws, n)
    # Expand dims for broadcasting:
    # r_i: (draws, n, 1) -> ranks of driver i
    # r_j: (draws, 1, n) -> ranks of driver j
    r_i = ranks[:, :, np.newaxis]
    r_j = ranks[:, np.newaxis, :]

    # ahead[d, i, j] = 1 if driver i finished ahead of driver j in draw d
    ahead = (r_i < r_j).astype(float)
    pairwise = ahead.sum(axis=0)

    # Convert counts to probabilities
    prob_matrix = counts / draws
    
    # Calculate mean position (1-indexed)
    positions = np.arange(1, n + 1)
    mean_pos = (prob_matrix * positions).sum(axis=1)
    
    # Pairwise probability matrix
    pairwise_prob = pairwise / draws
    np.fill_diagonal(pairwise_prob, 0.5)
    
    return prob_matrix, mean_pos, pairwise_prob
