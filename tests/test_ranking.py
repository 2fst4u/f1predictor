"""Tests for ranking utilities."""
import pytest
import numpy as np
from unittest.mock import patch
from f1pred.ranking import plackett_luce_scores


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

def test_pl_all_non_finite():
    # Covers line 25: `fill = 0.0` when no finite elements exist
    scores = np.array([np.nan, np.inf])
    probs = plackett_luce_scores(scores)
    assert np.allclose(probs, 0.5)

def test_pl_denom_invalid():
    # Covers line 44: `return np.full(n, 1.0 / n)` when denom is invalid
    scores = np.array([1.0, 2.0])
    with patch('numpy.exp', return_value=np.array([np.nan, np.nan])):
        probs = plackett_luce_scores(scores)
        assert np.allclose(probs, [0.5, 0.5])
