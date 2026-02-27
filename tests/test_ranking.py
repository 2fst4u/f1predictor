"""Tests for ranking utilities."""
import pytest
import numpy as np
from f1pred.ranking import plackett_luce_scores, rank_from_pace


# ---------------------------------------------------------------------------
# plackett_luce_scores
# ---------------------------------------------------------------------------
def test_pl_sums_to_one():
    scores = np.array([10.0, 5.0, 1.0])
    probs = plackett_luce_scores(scores)
    assert probs.sum() == pytest.approx(1.0)


def test_pl_higher_score_higher_prob():
    scores = np.array([10.0, 5.0, 1.0])
    probs = plackett_luce_scores(scores)
    assert probs[0] > probs[1] > probs[2]


def test_pl_empty():
    result = plackett_luce_scores(np.array([]))
    assert len(result) == 0


def test_pl_identical_scores():
    scores = np.array([5.0, 5.0, 5.0])
    probs = plackett_luce_scores(scores)
    # Should return uniform
    assert np.allclose(probs, 1 / 3)


def test_pl_nan_inf():
    scores = np.array([np.nan, 5.0, np.inf])
    probs = plackett_luce_scores(scores)
    assert probs.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(probs))


def test_pl_negative_temperature():
    scores = np.array([10.0, 5.0, 1.0])
    probs = plackett_luce_scores(scores, temperature=-1.0)
    # Should coerce to small positive temperature and still work
    assert probs.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# rank_from_pace
# ---------------------------------------------------------------------------
def test_rank_deterministic():
    pace = np.array([3.0, 1.0, 2.0])
    order = rank_from_pace(pace, noise_sd=0)
    # Sorted ascending: index 1 (1.0), 2 (2.0), 0 (3.0)
    assert list(order) == [1, 2, 0]


def test_rank_empty():
    result = rank_from_pace(np.array([]), noise_sd=0)
    assert len(result) == 0


def test_rank_reproducible():
    pace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    order1 = rank_from_pace(pace, noise_sd=0.5, random_state=42)
    order2 = rank_from_pace(pace, noise_sd=0.5, random_state=42)
    assert np.array_equal(order1, order2)


def test_rank_all_same_pace():
    pace = np.array([1.0, 1.0, 1.0])
    order = rank_from_pace(pace, noise_sd=0)
    # Stable sort should preserve original order
    assert list(order) == [0, 1, 2]
