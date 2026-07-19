"""Tests for the canonical FastF1 official (penalty-adjusted) starting grid parse."""
import numpy as np
import pandas as pd

from f1pred.features import official_grid_from_classification


def _roster():
    return pd.DataFrame([
        {"driverId": "max_verstappen", "code": "VER"},
        {"driverId": "leclerc", "code": "LEC"},
        {"driverId": "hamilton", "code": "HAM"},
    ])


def test_reads_gridposition_and_maps_by_code():
    # LEC qualified ahead but a penalty drops it behind HAM: grid reflects that.
    cls = pd.DataFrame([
        {"Abbreviation": "VER", "GridPosition": 1.0},
        {"Abbreviation": "HAM", "GridPosition": 2.0},
        {"Abbreviation": "LEC", "GridPosition": 3.0},
    ])
    out = official_grid_from_classification(cls, _roster())
    got = dict(zip(out["driverId"], out["grid"]))
    assert got == {"max_verstappen": 1, "hamilton": 2, "leclerc": 3}
    # grid must come back as plain ints, not numpy floats
    assert all(isinstance(g, int) for g in out["grid"])


def test_zero_and_nan_gridpositions_are_skipped():
    # GridPosition 0 (pit lane / not published) and NaN are treated as unknown.
    cls = pd.DataFrame([
        {"Abbreviation": "VER", "GridPosition": 0.0},
        {"Abbreviation": "HAM", "GridPosition": np.nan},
        {"Abbreviation": "LEC", "GridPosition": 4.0},
    ])
    out = official_grid_from_classification(cls, _roster())
    assert dict(zip(out["driverId"], out["grid"])) == {"leclerc": 4}


def test_empty_when_grid_not_published():
    cls = pd.DataFrame([
        {"Abbreviation": "VER", "GridPosition": 0.0},
        {"Abbreviation": "HAM", "GridPosition": 0.0},
    ])
    out = official_grid_from_classification(cls, _roster())
    assert out.empty
    assert list(out.columns) == ["driverId", "grid"]


def test_missing_gridposition_column_returns_empty():
    cls = pd.DataFrame([{"Abbreviation": "VER", "Position": 1.0}])
    assert official_grid_from_classification(cls, _roster()).empty


def test_none_and_empty_inputs_return_empty():
    assert official_grid_from_classification(None, _roster()).empty
    assert official_grid_from_classification(pd.DataFrame(), _roster()).empty


def test_driver_not_in_roster_is_skipped():
    cls = pd.DataFrame([
        {"Abbreviation": "VER", "GridPosition": 1.0},
        {"Abbreviation": "XYZ", "GridPosition": 2.0},  # reservist not on roster
    ])
    out = official_grid_from_classification(cls, _roster())
    assert dict(zip(out["driverId"], out["grid"])) == {"max_verstappen": 1}


def test_roster_without_code_column_returns_empty():
    cls = pd.DataFrame([{"Abbreviation": "VER", "GridPosition": 1.0}])
    roster = pd.DataFrame([{"driverId": "max_verstappen"}])
    assert official_grid_from_classification(cls, roster).empty
