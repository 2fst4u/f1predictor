"""End-to-end exercise of CalibrationManager.run_calibration.

The optimisation core in calibrate.py (run_calibration + the nested objective
function — ~570 lines) was previously untested: no test ever invoked
run_calibration, so the module sat at ~22% coverage and a latent NameError
(``res.fun`` instead of ``best_res.fun`` at the point where the optimised
objective score is stored) went undetected because it was swallowed by the
broad ``except Exception`` around the whole routine.

These tests drive the real objective function and optimiser over a small
synthetic dataset, mocking only the expensive feature-building and
GBM-training steps and capping the L-BFGS-B iteration count (the goal is to
execute the per-event z-scoring, ensemble blending, Spearman/podium loss,
regularisation, and the final weight-pack + persistence path — not to verify
optimisation quality). The full calibration is run once per module via a
shared fixture to keep the suite fast.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import scipy.optimize as _sciopt

from f1pred.calibrate import CalibrationManager


# --- minimal config object -------------------------------------------------

class _CalCfg:
    enabled = True
    lookback_window_days = 365
    frequency_hours = 24

    def __init__(self, weights_file):
        self.weights_file = weights_file


class _RecencyCfg:
    base = 120.0
    team = 240.0


class _ModellingCfg:
    recency_half_life_days = _RecencyCfg()


class _Cfg:
    def __init__(self, weights_file):
        self.calibration = _CalCfg(weights_file)
        self.modelling = _ModellingCfg()


# --- synthetic history -----------------------------------------------------

_DRIVERS = [f"drv_{i}" for i in range(8)]
_TEAMS = [f"team_{i // 2}" for i in range(8)]


def _event_rows(season, rnd, date, session):
    """One event worth of finishing-order rows for all 8 drivers."""
    rows = []
    for pos, (drv, team) in enumerate(zip(_DRIVERS, _TEAMS), start=1):
        rows.append({
            "season": season,
            "round": rnd,
            "date": date,
            "session": session,
            "driverId": drv,
            "constructorId": team,
            "circuitId": f"circuit_{rnd}",
            "position": float(pos),
            "qpos": float(pos),
            "grid": float(pos),
            "points": max(0.0, 26.0 - pos),
            "status": "Finished",
        })
    return rows


def _make_history(now):
    """Older training races (pre-window) + recent calibration races/quali."""
    rows = []
    # Training data well before the lookback window (~2 years back).
    for r in range(1, 11):
        d = now - timedelta(days=730 - r * 20)
        rows += _event_rows(2024, r, d, "race")
        rows += _event_rows(2024, r, d, "qualifying")
    # Calibration data inside the 365-day window.
    for r in range(1, 7):
        d = now - timedelta(days=300 - r * 40)
        rows += _event_rows(2025, r, d, "race")
        rows += _event_rows(2025, r, d, "qualifying")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def _fake_features(*args, **kwargs):
    """Stand-in for build_session_features: returns (X, meta, roster).

    Signature in source: build_session_features(jc, om, season, round,
    session, date, cfg).
    """
    X = pd.DataFrame({
        "driverId": _DRIVERS,
        "constructorId": _TEAMS,
        "form_index": np.linspace(-1.0, 1.0, len(_DRIVERS)),
        "team_form_index": np.linspace(-0.5, 0.5, len(_DRIVERS)),
        "grid": np.arange(1, len(_DRIVERS) + 1, dtype=float),
    })
    return X, {"raceName": "Synthetic GP"}, X[["driverId", "constructorId"]]


class _FakePipe:
    """GBM stand-in whose prediction tracks grid (so it correlates with truth)."""

    def predict(self, X):
        if isinstance(X, pd.DataFrame) and "grid" in X.columns:
            return np.asarray(X["grid"], dtype=float)
        return np.zeros(len(X), dtype=float)


def _fake_train_pace_model(X_train, session, cfg, *a, **k):
    features = ["grid", "form_index", "team_form_index"]
    return _FakePipe(), None, features, None


_real_minimize = _sciopt.minimize


def _fast_minimize(fun, x0, **kwargs):
    """Wrap scipy.optimize.minimize with a tiny iteration cap for speed.

    The default calibration runs L-BFGS-B with maxiter=800 over 26 params x 5
    starts, which takes minutes. The tests only need the code path executed,
    so we cap iterations hard. (run_calibration uses a local
    ``from scipy.optimize import minimize`` that resolves the patched attr.)
    """
    options = {**kwargs.get("options", {}), "maxiter": 3}
    kwargs["options"] = options
    return _real_minimize(fun, x0, **kwargs)


def _run_full_calibration(cm):
    now = datetime.now(timezone.utc)
    hist = _make_history(now)
    with patch("f1pred.features.build_session_features", side_effect=_fake_features), \
         patch("f1pred.models.train_pace_model", side_effect=_fake_train_pace_model), \
         patch("scipy.optimize.minimize", side_effect=_fast_minimize):
        cm.run_calibration(jc=object(), om=object(), history_df=hist)
    return cm


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory):
    """Run a full (iteration-capped) calibration once and share the result."""
    weights_file = tmp_path_factory.mktemp("cal") / "weights.json"
    cm = CalibrationManager(_Cfg(str(weights_file)))
    return _run_full_calibration(cm)


def test_run_calibration_stores_finite_objective_score(calibrated):
    """Regression guard for the res.fun NameError.

    On a successful run the optimised objective value must be written into
    current_weights. Before the fix this raised NameError (res undefined),
    was swallowed by the broad except, and objective_score was never set.
    """
    assert "objective_score" in calibrated.current_weights, (
        "run_calibration did not store objective_score — the final weight-write "
        "path failed (regression of the res.fun/best_res.fun bug)."
    )
    assert np.isfinite(calibrated.current_weights["objective_score"])


def test_run_calibration_persists_weights_and_metadata(calibrated):
    """A successful run saves weights with version + last_race_id to disk."""
    assert Path(calibrated.weights_file).exists()
    # last_race_id is set from the final event in the lookback window.
    assert calibrated.last_race_id == "2025_6"

    reloaded = CalibrationManager(_Cfg(str(calibrated.weights_file)))
    data = reloaded.load_weights()
    assert "calibration_version" in data
    assert "calibration_timestamp" in data
    # All six tunable groups survive the optimisation + round-trip.
    for group in ("ensemble", "blending", "dnf", "simulation", "recency", "elo"):
        assert group in data


def test_run_calibration_keeps_ensemble_weights_normalised(calibrated):
    """Optimised race + qualifying ensemble weights are each renormalised to ~1."""
    ens = calibrated.current_weights["ensemble"]
    race_total = ens["w_gbm"] + ens["w_elo"] + ens["w_bt"] + ens["w_mixed"]
    quali_total = (ens["w_gbm_quali"] + ens["w_elo_quali"]
                   + ens["w_bt_quali"] + ens["w_mixed_quali"])
    assert race_total == pytest.approx(1.0, abs=1e-5)
    assert quali_total == pytest.approx(1.0, abs=1e-5)


def test_run_calibration_aborts_cleanly_with_no_recent_races(tmp_path):
    """No races in the lookback window -> early return, defaults untouched."""
    cm = CalibrationManager(_Cfg(str(tmp_path / "weights.json")))
    now = datetime.now(timezone.utc)
    old_rows = []
    for r in range(1, 4):
        old_rows += _event_rows(2020, r, now - timedelta(days=1500 + r), "race")
    hist = pd.DataFrame(old_rows)
    hist["date"] = pd.to_datetime(hist["date"], utc=True)

    with patch("f1pred.features.build_session_features", side_effect=_fake_features), \
         patch("f1pred.models.train_pace_model", side_effect=_fake_train_pace_model), \
         patch("scipy.optimize.minimize", side_effect=_fast_minimize):
        cm.run_calibration(jc=object(), om=object(), history_df=hist)

    # Aborted before optimisation: no objective score recorded, file not written.
    assert "objective_score" not in cm.current_weights
    assert not Path(cm.weights_file).exists()
