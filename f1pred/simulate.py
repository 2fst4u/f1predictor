"""Monte Carlo simulation for F1 race outcomes.

This module simulates race finishes by adding stochastic noise to pace
predictions and applying DNF probabilities.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

__all__ = ["simulate_grid", "noise_sigma"]

# Scale linking noise_factor to the pace standard deviation.  Kept as a module
# constant so the calibration objective can reproduce the exact same sigma
# analytically (see calibrate.py).
NOISE_STD_MULTIPLIER = 3.0


def noise_sigma(pace_index: np.ndarray, noise_factor: float, min_noise: float) -> float:
    """Per-driver Gaussian noise std used by the simulation.

    Based on the standard deviation of the pace spread (robust to a single
    outlier, unlike the previous max-min range) with a floor of min_noise.
    """
    pace_std = float(np.std(pace_index)) if len(pace_index) else 0.0
    if not np.isfinite(pace_std) or pace_std < 1e-6:
        pace_std = 0.1
    return max(pace_std * float(noise_factor) * NOISE_STD_MULTIPLIER, float(min_noise))


def simulate_grid(
    pace_index: np.ndarray,
    dnf_prob: np.ndarray,
    draws: int = 5000,
    random_seed: int = 42,
    noise_factor: float = 0.15,
    min_noise: float = 0.05,
    max_penalty_base: float = 20.0,
    compute_pairwise: bool = True,
    team_codes: Optional[np.ndarray] = None,
    team_correlation: float = 0.0,
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
        team_codes: Optional integer team index per driver.  When given together
                    with team_correlation > 0, a shared per-team noise component
                    is mixed in so teammates' outcomes are correlated (car
                    performance on the day is a team-level random effect).
        team_correlation: Fraction of noise variance shared within a team (0-1).
    """
    rng = np.random.RandomState(random_seed)

    # Input sanitization: ensure finite values
    pace_index = np.asarray(pace_index, dtype=float)
    dnf_prob = np.asarray(dnf_prob, dtype=float)

    n = len(pace_index)
    
    if n == 0:
        return np.array([]).reshape(0, 0), np.array([]), np.array([]).reshape(0, 0)

    # Replace non-finite pace values with the median of finite values
    finite_mask = np.isfinite(pace_index)
    if not finite_mask.all():
        fill = float(np.median(pace_index[finite_mask])) if finite_mask.any() else 0.0
        pace_index = np.where(finite_mask, pace_index, fill)

    # Clamp DNF probabilities to [0, 1] and replace non-finite with 0
    dnf_prob = np.where(np.isfinite(dnf_prob), dnf_prob, 0.0)
    dnf_prob = np.clip(dnf_prob, 0.0, 1.0)

    if len(dnf_prob) != n:
        dnf_prob = np.full(n, 0.08, dtype=float)

    # Calculate noise scale based on pace spread (std-based; robust to outliers)
    noise_scale = noise_sigma(pace_index, noise_factor, min_noise)

    # DNF penalty - move driver to back of grid
    max_penalty = abs(np.max(pace_index)) + abs(np.min(pace_index)) + max_penalty_base

    # ---------------------------------------------------------
    # Vectorized Simulation
    # ---------------------------------------------------------

    # 1. Generate all random events at once (Shape: draws x n)
    # ⚡ Bolt: standard_normal is slightly faster than normal(0, scale) due to less C overhead
    indiv_noise = rng.standard_normal((draws, n))

    rho = float(np.clip(team_correlation, 0.0, 1.0)) if team_correlation else 0.0
    if rho > 0.0 and team_codes is not None and len(team_codes) == n:
        # Mix a shared per-team component with the individual component so the
        # total variance stays noise_scale^2 while teammates are correlated.
        codes = np.asarray(team_codes, dtype=int)
        valid_codes = codes >= 0
        n_teams = int(codes.max()) + 1 if valid_codes.any() else 0
        if n_teams > 0:
            team_draws = rng.standard_normal((draws, n_teams))
            shared = np.where(valid_codes[None, :], team_draws[:, np.clip(codes, 0, None)], 0.0)
            mix = np.sqrt(rho) * shared + np.sqrt(1.0 - rho) * indiv_noise
            # Drivers without a team keep pure individual noise
            indiv_noise = np.where(valid_codes[None, :], mix, indiv_noise)

    noise = indiv_noise * noise_scale
    dnf_draws = rng.binomial(1, dnf_prob, size=(draws, n))

    # 2. Compute simulated pace for all draws
    # pace_index broadcasts to (draws, n)
    sim = pace_index + noise + dnf_draws * max_penalty

    # 3. Determine finishing order (indices of drivers sorted by sim value)
    # axis=1 sorts each draw independently
    order = np.argsort(sim, axis=1)

    # 4. Compute Counts Matrix (n x n)
    # counts[driver_i, pos_j] = number of times driver i finished at pos j

    # Count how many times each driver finished at each position
    # Loop over positions (columns of order) and use bincount on driver indices
    # ⚡ Bolt: A list comprehension assigned to array and transposed is faster
    # than assigning column-by-column in a pre-allocated array.
    counts = np.array(
        [np.bincount(order[:, p], minlength=n) for p in range(n)],
        dtype=float
    ).T

    # 5. Compute Pairwise Matrix (n x n)
    # pairwise[i, j] = count of draws where driver i finished ahead of driver j

    pairwise_prob = np.array([])

    if compute_pairwise:
        # Compute ranks: ranks[d, i] is the position (0..n-1) of driver i in draw d
        # Optimization: use int16 to reduce memory bandwidth during broadcasting (N <= 32767)
        ranks = np.empty((draws, n), dtype=np.int16)

        # Create row indices: [[0], [1], ..., [draws-1]]
        row_indices = np.arange(draws)[:, None]

        # Use advanced indexing to invert the permutation
        # If order[d, p] = driver_idx, then ranks[d, driver_idx] = p
        ranks[row_indices, order] = np.arange(n, dtype=np.int16)

        # Pairwise comparison via 2D vectorization over the upper triangle
        # This prevents allocating a massive (draws, n, n) array, saving memory
        # and reducing computation time by ~40% for typical grid sizes
        pairwise = np.zeros((n, n), dtype=float)
        for i in range(n - 1):
            # Vectorized comparison for upper triangle
            wins = np.sum(ranks[:, i:i+1] < ranks[:, i+1:], axis=0)
            pairwise[i, i+1:] = wins
            pairwise[i+1:, i] = draws - wins

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
