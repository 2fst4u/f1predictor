from __future__ import annotations
from typing import Tuple
import numpy as np

def simulate_grid(pace_index: np.ndarray, dnf_prob: np.ndarray, draws: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo simulation: for each draw, add noise to pace, apply DNFs, and derive finishing order.

    Returns:
      - prob_matrix: [n_drivers, n_drivers] probability of finishing at each position (1..N)
      - mean_pos: expected finishing position
      - pairwise_matrix: [n_drivers, n_drivers] where entry (i,j) = P(i finishes ahead of j)
    """
    n = len(pace_index)
    counts = np.zeros((n, n), dtype=float)
    pairwise = np.zeros((n, n), dtype=float)

    noise_scale = np.std(pace_index) * 0.15 + 0.05
    max_penalty = np.max(pace_index) + 10.0

    for _ in range(draws):
        noisy = pace_index + np.random.normal(0.0, noise_scale, size=n)
        dnf_draw = np.random.binomial(1, dnf_prob, size=n)
        sim = noisy + dnf_draw * max_penalty

        order = np.argsort(sim)
        for pos, idx in enumerate(order):
            counts[idx, pos] += 1.0

        rank = np.empty(n, dtype=int)
        rank[order] = np.arange(n)
        r_i = rank.reshape(-1, 1)
        r_j = rank.reshape(1, -1)
        ahead = (r_i < r_j).astype(float)
        pairwise += ahead

    prob_matrix = counts / draws
    mean_pos = (prob_matrix * (np.arange(n) + 1)).sum(axis=1)
    pairwise_prob = pairwise / draws
    np.fill_diagonal(pairwise_prob, 0.5)
    return prob_matrix, mean_pos, pairwise_prob