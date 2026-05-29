"""Tests for the qualifying weight path and z-normalisation in combine_pace.

The audit found that combine_pace was only ever exercised with the default
race weights: the qualifying branch (session_type in {"qualifying",
"sprint_qualifying"}), which selects the dedicated *_quali weights — the whole
reason that weight set exists — had no coverage, nor did the zero-variance /
zero-weight fallbacks.
"""
import numpy as np
import pytest

from f1pred.ensemble import combine_pace, EnsembleConfig


@pytest.fixture
def cfg():
    return EnsembleConfig()


def _ordering(arr):
    return np.argsort(np.asarray(arr)).tolist()


def test_race_session_uses_race_weights(cfg):
    """With race weights favouring Elo, the blended order follows Elo."""
    cfg.w_gbm, cfg.w_elo, cfg.w_bt, cfg.w_mixed = 0.0, 1.0, 0.0, 0.0
    cfg.w_gbm_quali, cfg.w_elo_quali, cfg.w_bt_quali, cfg.w_mixed_quali = 1.0, 0.0, 0.0, 0.0

    gbm = np.array([1.0, 2.0, 3.0, 4.0])      # ascending
    elo = np.array([4.0, 3.0, 2.0, 1.0])      # descending
    flat = np.array([5.0, 5.0, 5.0, 5.0])     # constant -> z == 0

    out = combine_pace(gbm, elo, flat, flat, cfg, session_type="race")
    assert _ordering(out) == _ordering(elo)


@pytest.mark.parametrize("session", ["qualifying", "sprint_qualifying"])
def test_quali_sessions_use_quali_weights(cfg, session):
    """Qualifying/sprint-qualifying select the *_quali weights (here: GBM)."""
    cfg.w_gbm, cfg.w_elo, cfg.w_bt, cfg.w_mixed = 0.0, 1.0, 0.0, 0.0
    cfg.w_gbm_quali, cfg.w_elo_quali, cfg.w_bt_quali, cfg.w_mixed_quali = 1.0, 0.0, 0.0, 0.0

    gbm = np.array([1.0, 2.0, 3.0, 4.0])      # ascending
    elo = np.array([4.0, 3.0, 2.0, 1.0])      # descending
    flat = np.array([5.0, 5.0, 5.0, 5.0])

    out = combine_pace(gbm, elo, flat, flat, cfg, session_type=session)
    # Order follows GBM (quali weight), NOT Elo (race weight).
    assert _ordering(out) == _ordering(gbm)
    assert _ordering(out) != _ordering(elo)


def test_constant_component_zero_variance_fallback(cfg):
    """A component with ~zero variance contributes zeros, not NaN/inf.

    combine_pace adds a deterministic tie-breaking jitter on the order of 1e-6,
    so the output is zero up to that jitter rather than exactly zero.
    """
    cfg.w_gbm, cfg.w_elo, cfg.w_bt, cfg.w_mixed = 1.0, 0.0, 0.0, 0.0
    const = np.array([7.0, 7.0, 7.0, 7.0])
    out = combine_pace(const, const, const, const, cfg, session_type="race")
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) < 1e-3  # only the ~1e-6 jitter remains


def test_nonpositive_weight_sum_falls_back_to_equal_weights(cfg):
    """All-zero weights must not divide by zero; output stays finite."""
    cfg.w_gbm = cfg.w_elo = cfg.w_bt = cfg.w_mixed = 0.0
    gbm = np.array([1.0, 2.0, 3.0, 4.0])
    elo = np.array([4.0, 3.0, 2.0, 1.0])
    bt = np.array([1.0, 1.0, 2.0, 2.0])
    mixed = np.array([2.0, 1.0, 2.0, 1.0])
    out = combine_pace(gbm, elo, bt, mixed, cfg, session_type="race")
    assert len(out) == 4
    assert np.all(np.isfinite(out))


def test_blended_output_is_zero_centred_for_znormed_inputs(cfg):
    """The final blend is z-normalised, so it is mean-zero up to the tiny jitter."""
    cfg.w_gbm = cfg.w_elo = cfg.w_bt = cfg.w_mixed = 0.25
    rng = np.random.default_rng(0)
    a, b, c, d = (rng.standard_normal(10) for _ in range(4))
    out = combine_pace(a, b, c, d, cfg, session_type="race")
    # Mean is ~0 plus the mean of the deterministic jitter (~(n-1)/2 * 1e-6).
    assert abs(float(np.mean(out))) < 1e-3
    assert float(np.std(out)) == pytest.approx(1.0, abs=1e-2)
