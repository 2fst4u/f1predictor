"""Direct correctness tests for the feature-engineering functions in features.py.

The audit flagged features.py as the largest remaining coverage gap (~44%): most
of the per-driver index builders (form, sprint, qualifying, circuit proficiency,
grid/finish delta) and the raw Jolpica block parser had no direct correctness
tests — they were only exercised transitively (if at all) through the heavily
mocked build_session_features path, which never asserted on their numeric output.

These tests pin the actual maths. For a single historical event the
recency-weight cancels out of every weighted mean, so each index reduces to a
closed form we can assert exactly:

    form_index                 = -position
    sprint_form_index          = -position + points
    qualifying_form_index      = -qpos
    grid_finish_delta          =  grid - position
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta

from f1pred.features import (
    compute_form_indices,
    compute_sprint_form,
    compute_qualifying_form,
    compute_sprint_qualifying_form,
    compute_circuit_proficiency,
    compute_circuit_globals,
    compute_grid_finish_delta,
    _parse_races_block,
)

REF = datetime(2024, 1, 1, tzinfo=timezone.utc)
EVT = datetime(2023, 12, 1, tzinfo=timezone.utc)


def _index(df, driver, col):
    return float(df.loc[df["driverId"] == driver, col].iloc[0])


# ---------------------------------------------------------------------------
# compute_form_indices
# ---------------------------------------------------------------------------

def test_form_index_single_event_is_negative_position():
    """With one event the recency weight cancels: form_index == -position."""
    hist = pd.DataFrame({
        "session": ["race", "race"], "season": [2023, 2023], "round": [1, 1],
        "driverId": ["a", "b"], "position": [1.0, 2.0],
        "points": [25.0, 18.0], "date": [EVT, EVT],
    })
    out = compute_form_indices(hist, REF, half_life_days=365)
    assert _index(out, "a", "form_index") == pytest.approx(-1.0)
    assert _index(out, "b", "form_index") == pytest.approx(-2.0)
    # Better finisher must rank higher (closer to zero).
    assert _index(out, "a", "form_index") > _index(out, "b", "form_index")


def test_form_index_includes_sprint_sessions():
    """Sprint finishes feed the overall form index alongside races."""
    hist = pd.DataFrame({
        "session": ["sprint"], "season": [2023], "round": [1],
        "driverId": ["a"], "position": [3.0], "points": [6.0], "date": [EVT],
    })
    out = compute_form_indices(hist, REF, half_life_days=365)
    # pos_score only (sprint contributes -position to the overall index)
    assert _index(out, "a", "form_index") == pytest.approx(-3.0)


def test_form_index_empty_returns_named_columns():
    out = compute_form_indices(pd.DataFrame(), REF, half_life_days=365)
    assert list(out.columns) == ["driverId", "form_index"]
    assert out.empty


def test_form_index_boost_factor_reweights_current_season():
    """boost_factor > 1 pulls the index toward the current-season race result."""
    old = datetime(2022, 6, 1, tzinfo=timezone.utc)
    hist = pd.DataFrame({
        "session": ["race", "race"], "season": [2022, 2023], "round": [1, 1],
        "driverId": ["a", "a"], "position": [10.0, 2.0],
        "points": [1.0, 18.0], "date": [old, EVT],
    })
    base = compute_form_indices(hist, REF, half_life_days=365, current_season=2023, boost_factor=1.0)
    boosted = compute_form_indices(hist, REF, half_life_days=365, current_season=2023, boost_factor=5.0)
    # Current-season result is P2 (-2); boosting must move the blend toward it.
    assert _index(boosted, "a", "form_index") > _index(base, "a", "form_index")


# ---------------------------------------------------------------------------
# compute_sprint_form / compute_sprint_qualifying_form
# ---------------------------------------------------------------------------

def test_sprint_form_combines_position_and_points():
    hist = pd.DataFrame({
        "session": ["sprint", "sprint"], "season": [2023, 2023], "round": [1, 1],
        "driverId": ["a", "b"], "position": [1.0, 2.0],
        "points": [8.0, 7.0], "date": [EVT, EVT],
    })
    out = compute_sprint_form(hist, REF, half_life_days=365)
    # weighted_val = -position + points
    assert _index(out, "a", "sprint_form_index") == pytest.approx(7.0)
    assert _index(out, "b", "sprint_form_index") == pytest.approx(5.0)


def test_sprint_form_empty_when_no_sprint_sessions():
    hist = pd.DataFrame({
        "session": ["race"], "season": [2023], "round": [1],
        "driverId": ["a"], "position": [1.0], "points": [25.0], "date": [EVT],
    })
    out = compute_sprint_form(hist, REF, half_life_days=365)
    assert out.empty
    assert list(out.columns) == ["driverId", "sprint_form_index"]


def test_sprint_qualifying_form_is_negative_qpos():
    hist = pd.DataFrame({
        "session": ["sprint_qualifying", "sprint_qualifying"],
        "season": [2023, 2023], "round": [1, 1],
        "driverId": ["a", "b"], "qpos": [1.0, 6.0], "date": [EVT, EVT],
    })
    out = compute_sprint_qualifying_form(hist, REF, half_life_days=365)
    assert _index(out, "a", "sprint_qualifying_form_index") == pytest.approx(-1.0)
    assert _index(out, "b", "sprint_qualifying_form_index") == pytest.approx(-6.0)


def test_sprint_qualifying_form_ignores_plain_qualifying():
    hist = pd.DataFrame({
        "session": ["qualifying"], "season": [2023], "round": [1],
        "driverId": ["a"], "qpos": [1.0], "date": [EVT],
    })
    out = compute_sprint_qualifying_form(hist, REF, half_life_days=365)
    assert out.empty


# ---------------------------------------------------------------------------
# compute_qualifying_form
# ---------------------------------------------------------------------------

def test_qualifying_form_is_negative_qpos_and_spans_both_quali_types():
    hist = pd.DataFrame({
        "session": ["qualifying", "sprint_qualifying"], "season": [2023, 2023],
        "round": [1, 2], "driverId": ["a", "b"], "qpos": [1.0, 5.0],
        "date": [EVT, EVT],
    })
    out = compute_qualifying_form(hist, REF, half_life_days=365)
    assert _index(out, "a", "qualifying_form_index") == pytest.approx(-1.0)
    assert _index(out, "b", "qualifying_form_index") == pytest.approx(-5.0)


def test_qualifying_form_missing_qpos_column_returns_empty():
    hist = pd.DataFrame({
        "session": ["qualifying"], "season": [2023], "round": [1],
        "driverId": ["a"], "date": [EVT],
    })
    out = compute_qualifying_form(hist, REF, half_life_days=365)
    assert out.empty
    assert list(out.columns) == ["driverId", "qualifying_form_index"]


# ---------------------------------------------------------------------------
# compute_circuit_proficiency
# ---------------------------------------------------------------------------

def test_circuit_proficiency_experience_avg_pos_and_dnf_rate():
    """Three starts at one circuit: 2 finishes (P2, P4) and 1 DNF (accident)."""
    hist = pd.DataFrame({
        "session": ["race", "race", "race"],
        "season": [2022, 2023, 2023], "round": [1, 1, 2],
        "driverId": ["a", "a", "a"],
        "position": [2.0, 4.0, np.nan],
        "status": ["Finished", "Finished", "Accident"],
        "circuitId": ["monza", "monza", "monza"],
        "date": [datetime(2022, 1, 1, tzinfo=timezone.utc), EVT, EVT],
    })
    out = compute_circuit_proficiency(hist, "monza", REF)
    row = out[out["driverId"] == "a"].iloc[0]
    assert int(row["circuit_experience"]) == 3
    assert row["circuit_avg_pos"] == pytest.approx(3.0)      # (2 + 4) / 2 finishes
    assert row["circuit_dnf_rate"] == pytest.approx(1 / 3)   # 1 DNF of 3 starts


def test_circuit_proficiency_filters_to_target_circuit_before_ref_date():
    hist = pd.DataFrame({
        "session": ["race", "race", "race"],
        "season": [2023, 2023, 2099], "round": [1, 2, 1],
        "driverId": ["a", "a", "a"],
        "position": [1.0, 5.0, 1.0],
        "status": ["Finished", "Finished", "Finished"],
        "circuitId": ["monza", "spa", "monza"],
        "date": [EVT, EVT, datetime(2099, 1, 1, tzinfo=timezone.utc)],
    })
    out = compute_circuit_proficiency(hist, "monza", REF)
    # Only the single pre-ref-date Monza race counts (the 2099 one is filtered).
    row = out[out["driverId"] == "a"].iloc[0]
    assert int(row["circuit_experience"]) == 1
    assert row["circuit_avg_pos"] == pytest.approx(1.0)


def test_circuit_proficiency_missing_circuit_column_returns_empty():
    hist = pd.DataFrame({
        "session": ["race"], "season": [2023], "round": [1],
        "driverId": ["a"], "position": [1.0], "status": ["Finished"], "date": [EVT],
    })
    out = compute_circuit_proficiency(hist, "monza", REF)
    assert out.empty


def test_circuit_proficiency_blank_circuit_id_returns_empty():
    hist = pd.DataFrame({
        "session": ["race"], "season": [2023], "round": [1], "driverId": ["a"],
        "position": [1.0], "status": ["Finished"], "circuitId": ["monza"], "date": [EVT],
    })
    assert compute_circuit_proficiency(hist, "", REF).empty


# ---------------------------------------------------------------------------
# compute_circuit_globals
# ---------------------------------------------------------------------------

def test_circuit_globals_overtake_difficulty_and_dnf_rate():
    hist = pd.DataFrame({
        "session": ["race", "race", "race"],
        "season": [2023, 2023, 2023], "round": [1, 2, 3],
        "driverId": ["a", "b", "c"],
        "grid": [5.0, 6.0, np.nan],
        "position": [2.0, 4.0, np.nan],
        "status": ["Finished", "Finished", "Accident"],
        "circuitId": ["monza", "monza", "monza"],
        "date": [EVT, EVT, EVT],
    })
    g = compute_circuit_globals(hist, "monza", REF)
    # gains = (5-2)=3 and (6-4)=2 -> mean 2.5
    assert g["circuit_overtake_difficulty"] == pytest.approx(2.5)
    # 1 DNF of 3 classified rows
    assert g["global_circuit_dnf_rate"] == pytest.approx(1 / 3)


def test_circuit_globals_no_history_returns_neutral_defaults():
    g = compute_circuit_globals(pd.DataFrame(), "monza", REF)
    assert g == {"circuit_overtake_difficulty": 0.0, "global_circuit_dnf_rate": 0.08}


# ---------------------------------------------------------------------------
# compute_grid_finish_delta
# ---------------------------------------------------------------------------

def test_grid_finish_delta_is_grid_minus_position():
    hist = pd.DataFrame({
        "session": ["race", "race"], "season": [2023, 2023], "round": [1, 1],
        "driverId": ["a", "b"], "grid": [3.0, 1.0], "position": [1.0, 2.0],
        "date": [EVT, EVT],
    })
    out = compute_grid_finish_delta(hist, REF, half_life_days=365)
    assert _index(out, "a", "grid_finish_delta") == pytest.approx(2.0)   # gained 2
    assert _index(out, "b", "grid_finish_delta") == pytest.approx(-1.0)  # lost 1


def test_grid_finish_delta_ignores_non_race_sessions():
    hist = pd.DataFrame({
        "session": ["qualifying"], "season": [2023], "round": [1],
        "driverId": ["a"], "grid": [3.0], "position": [1.0], "date": [EVT],
    })
    out = compute_grid_finish_delta(hist, REF, half_life_days=365)
    assert out.empty
    assert list(out.columns) == ["driverId", "grid_finish_delta"]


# ---------------------------------------------------------------------------
# _parse_races_block
# ---------------------------------------------------------------------------

def _race_block():
    return [{
        "season": "2023", "round": "1", "date": "2023-03-05", "time": "14:00:00Z",
        "Circuit": {"circuitName": "Bahrain", "circuitId": "bahrain",
                    "Location": {"lat": "26.0", "long": "50.5"}},
        "Results": [{
            "Driver": {"driverId": "verstappen", "code": "VER"},
            "Constructor": {"constructorId": "red_bull"},
            "grid": "1", "position": "1", "status": "Finished", "points": "25",
            "FastestLap": {"rank": "1"},
        }],
    }]


def test_parse_races_block_race_row_normalisation():
    rows = _parse_races_block(_race_block(), "race", REF)
    assert len(rows) == 1
    row = rows[0]
    assert row["session"] == "race"
    assert row["season"] == 2023 and row["round"] == 1
    assert row["driverId"] == "verstappen"
    assert row["constructorId"] == "red_bull"
    assert row["grid"] == 1 and row["position"] == 1
    assert row["points"] == pytest.approx(25.0)
    assert row["lat"] == pytest.approx(26.0) and row["lon"] == pytest.approx(50.5)
    assert isinstance(row["date"], datetime) and row["date"].tzinfo is not None


def test_parse_races_block_qualifying_and_sprint_labels():
    qblock = [{
        "season": "2023", "round": "2", "date": "2023-03-12", "time": "13:00:00Z",
        "Circuit": {"circuitName": "Jeddah", "circuitId": "jeddah", "Location": {}},
        "QualifyingResults": [{
            "Driver": {"driverId": "leclerc", "code": "LEC"},
            "Constructor": {"constructorId": "ferrari"},
            "position": "1", "Q1": "1:28.0", "Q2": "1:27.5", "Q3": "1:27.0",
        }],
    }]
    qrows = _parse_races_block(qblock, "qualifying", REF)
    assert qrows[0]["session"] == "qualifying"
    assert qrows[0]["qpos"] == 1
    assert qrows[0]["q3"] == "1:27.0"

    sblock = [{
        "season": "2023", "round": "3", "date": "2023-04-29", "time": "15:00:00Z",
        "Circuit": {"circuitName": "Baku", "circuitId": "baku", "Location": {}},
        "SprintResults": [{
            "Driver": {"driverId": "perez", "code": "PER"},
            "Constructor": {"constructorId": "red_bull"}, "position": "2",
        }],
    }]
    srows = _parse_races_block(sblock, "sprint", REF)
    assert srows[0]["session"] == "sprint"
    assert srows[0]["position"] == 2


def test_parse_races_block_skips_rows_after_cutoff_and_without_round():
    block = [
        {"date": "2023-01-01"},  # no round -> skipped
        {  # future date relative to a tighter cutoff -> skipped
            "season": "2023", "round": "9", "date": "2023-12-31", "time": "00:00:00Z",
            "Circuit": {"Location": {}}, "Results": [],
        },
    ]
    cutoff = datetime(2023, 6, 1, tzinfo=timezone.utc)
    assert _parse_races_block(block, "race", cutoff) == []


def test_parse_races_block_malformed_date_falls_back_to_epoch():
    block = [{
        "season": "2023", "round": "1", "date": "not-a-date", "time": "nope",
        "Circuit": {"circuitId": "x", "Location": {}},
        "Results": [{"Driver": {"driverId": "a"}, "Constructor": {},
                     "position": "1", "status": "Finished"}],
    }]
    rows = _parse_races_block(block, "race", REF)
    # Epoch fallback (1970) is before REF, so the row survives with that date.
    assert len(rows) == 1
    assert rows[0]["date"].year == 1970
