"""Tests for ensemble models and pace combination."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from f1pred.ensemble import (
    EloModel,
    BradleyTerryModel,
    MixedEffectsLikeModel,
    EnsembleConfig,
    combine_pace,
    _safe_component,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def race_history():
    """Minimal historical race data for fitting ensemble models."""
    rows = []
    for season in [2023, 2024]:
        for rnd in range(1, 4):
            for pos, driver in enumerate(["d_alpha", "d_beta", "d_gamma"], start=1):
                rows.append({
                    "season": season,
                    "round": rnd,
                    "session": "race",
                    "date": datetime(season, 3 + rnd, 1, tzinfo=timezone.utc),
                    "driverId": driver,
                    "constructorId": f"team_{pos}",
                    "position": pos,
                    "points": [25, 18, 15][pos - 1],
                    "status": "Finished",
                })
    return pd.DataFrame(rows)


@pytest.fixture
def roster_X():
    """Feature matrix / roster for predictions."""
    return pd.DataFrame({
        "driverId": ["d_alpha", "d_beta", "d_gamma"],
        "constructorId": ["team_1", "team_2", "team_3"],
        "form_index": [1.0, 0.0, -1.0],
    })


# ---------------------------------------------------------------------------
# _safe_component
# ---------------------------------------------------------------------------
def test_safe_component_none():
    result = _safe_component(None, 5, "test")
    assert np.array_equal(result, np.zeros(5))


def test_safe_component_wrong_length():
    result = _safe_component(np.array([1.0, 2.0]), 5, "test")
    assert np.array_equal(result, np.zeros(5))


def test_safe_component_correct():
    arr = np.array([1.0, 2.0, 3.0])
    result = _safe_component(arr, 3, "test")
    assert np.array_equal(result, arr)


# ---------------------------------------------------------------------------
# EloModel
# ---------------------------------------------------------------------------
def test_elo_fit_empty():
    model = EloModel()
    model.fit(pd.DataFrame())
    assert model.ratings_ == {}


def test_elo_fit_and_predict(race_history, roster_X):
    model = EloModel().fit(race_history)

    # d_alpha always wins so should have highest rating
    assert model.ratings_["d_alpha"] > model.ratings_["d_gamma"]

    pace = model.predict(roster_X)
    assert len(pace) == 3
    # Lower pace = better. d_alpha (best) should have lowest pace
    assert pace[0] < pace[2]


def test_elo_predict_empty():
    model = EloModel()
    result = model.predict(pd.DataFrame())
    assert len(result) == 0


def test_elo_predict_no_driver_id():
    model = EloModel()
    X = pd.DataFrame({"foo": [1, 2]})
    result = model.predict(X)
    assert np.array_equal(result, np.zeros(2))


# ---------------------------------------------------------------------------
# BradleyTerryModel
# ---------------------------------------------------------------------------
def test_bt_fit_empty():
    model = BradleyTerryModel()
    model.fit(pd.DataFrame())
    assert model.strength_ == {}


def test_bt_fit_and_predict(race_history, roster_X):
    model = BradleyTerryModel().fit(race_history)

    # d_alpha consistently wins → higher strength
    assert model.strength_["d_alpha"] > model.strength_["d_gamma"]

    pace = model.predict(roster_X)
    assert len(pace) == 3
    # Lower pace = better
    assert pace[0] < pace[2]


def test_bt_predict_unknown_driver():
    model = BradleyTerryModel()
    X = pd.DataFrame({"driverId": ["unknown"]})
    pace = model.predict(X)
    assert len(pace) == 1
    assert np.isfinite(pace[0])


# ---------------------------------------------------------------------------
# MixedEffectsLikeModel
# ---------------------------------------------------------------------------
def test_mixed_fit_empty():
    model = MixedEffectsLikeModel()
    model.fit(pd.DataFrame())
    assert model.driver_effect_ == {}
    assert model.team_effect_ == {}


def test_mixed_fit_and_predict(race_history, roster_X):
    model = MixedEffectsLikeModel().fit(race_history)

    pace = model.predict(roster_X)
    assert len(pace) == 3
    # d_alpha consistently wins → lower pace (faster)
    assert pace[0] < pace[2]


# ---------------------------------------------------------------------------
# combine_pace
# ---------------------------------------------------------------------------
def test_combine_pace_basic():
    gbm = np.array([1.0, 2.0, 3.0])
    result = combine_pace(gbm, None, None, None)
    assert len(result) == 3
    assert np.all(np.isfinite(result))


def test_combine_pace_all_components():
    n = 5
    gbm = np.arange(n, dtype=float)
    elo = np.arange(n, dtype=float) * 0.5
    bt = np.arange(n, dtype=float) * 0.3
    mixed = np.arange(n, dtype=float) * 0.2

    cfg = EnsembleConfig(w_gbm=0.4, w_elo=0.2, w_bt=0.2, w_mixed=0.2)
    result = combine_pace(gbm, elo, bt, mixed, cfg=cfg)
    assert len(result) == n
    assert np.all(np.isfinite(result))
    # Output should preserve ordering (lower input → lower output)
    assert result[0] < result[-1]


def test_combine_pace_none_gbm():
    result = combine_pace(None, None, None, None)
    assert len(result) == 0


def test_combine_pace_empty():
    result = combine_pace(np.array([]), None, None, None)
    assert len(result) == 0


def test_combine_pace_wrong_length_component():
    gbm = np.array([1.0, 2.0, 3.0])
    elo = np.array([1.0])  # Wrong length
    result = combine_pace(gbm, elo, None, None)
    assert len(result) == 3
    assert np.all(np.isfinite(result))
