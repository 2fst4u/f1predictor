"""Monte Carlo simulation for F1 race outcomes.

This module simulates race finishes by adding stochastic noise to pace
predictions and applying DNF probabilities.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

__all__ = ["simulate_grid"]


def simulate_grid(
    pace_index: np.ndarray,
    dnf_prob: np.ndarray,
    draws: int = 5000,
    random_seed: int = 42,
    noise_factor: float = 0.15,
    min_noise: float = 0.05,
    max_penalty_base: float = 20.0,
    compute_pairwise: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo simulation: for each draw, add noise to pace, apply DNFs, and derive finishing order.

    Args:
        pace_index: Lower values = faster driver
        dnf_prob: Probability of DNF for each driver (0 to 1)
        draws: Number of simulation iterations
        random_seed: For reproducibility
        noise_factor: Multiplier for pace spread to determine noise scale
        min_noise: Minimum noise standard deviation
        max_penalty_base: Base penalty added to max pace for DNF drivers
        compute_pairwise: Whether to compute the O(N^2) pairwise probability matrix.
                          Defaults to True for backward compatibility.
    """
    rng = np.random.RandomState(random_seed)
    n = len(pace_index)
    
    if n == 0:
        return np.array([]).reshape(0, 0), np.array([]), np.array([]).reshape(0, 0)
    
    # Calculate noise scale based on pace spread
    pace_std = float(np.std(pace_index))
    pace_range = float(np.ptp(pace_index))  # max - min
    
    if pace_range > 0.01:
        noise_scale = pace_range * noise_factor
    else:
        noise_scale = 0.1
    
    noise_scale = max(noise_scale, min_noise)
    
    # DNF penalty - move driver to back of grid
    max_penalty = abs(np.max(pace_index)) + abs(np.min(pace_index)) + max_penalty_base

    # ---------------------------------------------------------
    # Vectorized Simulation
    # ---------------------------------------------------------

    # 1. Generate all random events at once (Shape: draws x n)
    noise = rng.normal(0.0, noise_scale, size=(draws, n))
    dnf_draws = rng.binomial(1, dnf_prob, size=(draws, n))

    # 2. Compute simulated pace for all draws
    # pace_index broadcasts to (draws, n)
    sim = pace_index + noise + dnf_draws * max_penalty

    # 3. Determine finishing order (indices of drivers sorted by sim value)
    # axis=1 sorts each draw independently
    order = np.argsort(sim, axis=1)

    # 4. Compute Counts Matrix (n x n)
    # counts[driver_i, pos_j] = number of times driver i finished at pos j

    counts = np.zeros((n, n), dtype=float)

    # Flatten order to get driver indices for all (draw, position) pairs
    drivers_flat = order.flatten()

    # Corresponding positions: [0, 1, ..., n-1] repeated 'draws' times
    positions_flat = np.tile(np.arange(n), draws)

    # Accumulate counts
    np.add.at(counts, (drivers_flat, positions_flat), 1.0)

    # 5. Compute Pairwise Matrix (n x n)
    # pairwise[i, j] = count of draws where driver i finished ahead of driver j

    pairwise_prob = np.array([])

    if compute_pairwise:
        # Compute ranks: ranks[d, i] is the position (0..n-1) of driver i in draw d
        ranks = np.empty((draws, n), dtype=int)

        # Create row indices: [[0], [1], ..., [draws-1]]
        row_indices = np.arange(draws)[:, None]

        # Use advanced indexing to invert the permutation
        # If order[d, p] = driver_idx, then ranks[d, driver_idx] = p
        ranks[row_indices, order] = np.arange(n)

        # Pairwise comparison via broadcasting
        # r_i shape: (draws, n, 1)
        # r_j shape: (draws, 1, n)
        r_i = ranks[:, :, None]
        r_j = ranks[:, None, :]

        # Sum boolean comparisons across draws
        pairwise = np.sum(r_i < r_j, axis=0, dtype=float)

        # Pairwise probability matrix
        pairwise_prob = pairwise / draws
        np.fill_diagonal(pairwise_prob, 0.5)

    # ---------------------------------------------------------
    # Post-processing (same as before)
    # ---------------------------------------------------------

    # Convert counts to probabilities
    prob_matrix = counts / draws
    
    # Calculate mean position (1-indexed)
    positions = np.arange(1, n + 1)
    mean_pos = (prob_matrix * positions).sum(axis=1)
    
    return prob_matrix, mean_pos, pairwise_prob
