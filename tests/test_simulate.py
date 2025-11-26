"""Tests for Monte Carlo simulation."""
import numpy as np
from f1pred.simulate import simulate_grid


def test_simulate_grid_basic():
    """Test basic simulation functionality."""
    n_drivers = 20
    pace_index = np.random.randn(n_drivers)
    dnf_prob = np.full(n_drivers, 0.1)

    prob_matrix, mean_pos, pairwise = simulate_grid(pace_index, dnf_prob, draws=1000)

    assert prob_matrix.shape == (n_drivers, n_drivers)
    assert len(mean_pos) == n_drivers
    assert pairwise.shape == (n_drivers, n_drivers)


def test_simulate_grid_probabilities_sum_to_one():
    """Test that position probabilities sum to 1 for each driver."""
    n_drivers = 20
    pace_index = np.random.randn(n_drivers)
    dnf_prob = np.full(n_drivers, 0.1)

    prob_matrix, _, _ = simulate_grid(pace_index, dnf_prob, draws=1000)

    # Each driver's probabilities across positions should sum to ~1
    row_sums = prob_matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=0.01)


def test_simulate_grid_faster_drivers_win_more():
    """Test that faster drivers (lower pace index) win more often."""
    n_drivers = 20
    # Create clear pace hierarchy
    pace_index = np.arange(n_drivers, dtype=float)
    dnf_prob = np.zeros(n_drivers)

    prob_matrix, mean_pos, _ = simulate_grid(pace_index, dnf_prob, draws=5000)

    # Best driver (pace_index=0) should have highest win probability
    win_probs = prob_matrix[:, 0]
    best_driver_idx = np.argmin(pace_index)
    assert win_probs[best_driver_idx] == np.max(win_probs)

    # Mean positions should correlate with pace
    # Lower pace should mean lower mean position
    assert mean_pos[0] < mean_pos[-1]


def test_simulate_grid_dnf_affects_results():
    """Test that DNF probability affects finishing positions."""
    n_drivers = 20
    pace_index = np.zeros(n_drivers)

    # One driver with high DNF probability
    dnf_prob_low = np.full(n_drivers, 0.01)
    dnf_prob_high = dnf_prob_low.copy()
    dnf_prob_high[0] = 0.9

    _, mean_pos_low, _ = simulate_grid(pace_index, dnf_prob_low, draws=5000)
    _, mean_pos_high, _ = simulate_grid(pace_index, dnf_prob_high, draws=5000)

    # Driver with high DNF prob should have worse mean position
    assert mean_pos_high[0] > mean_pos_low[0]
