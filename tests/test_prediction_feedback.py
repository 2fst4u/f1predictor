"""Regression tests for the intra-weekend prediction feedback loop.

Predictions for sessions that have not run yet must never re-enter the
feature-building history as if they were observed results.  Dated the same
day as the session being predicted, such rows receive the maximum recency
weight multiplied by the calibrated current-season boost, so they dominate
every form index and create a self-reinforcing feedback loop: tiny input
changes (e.g. an hourly weather refresh) amplify into huge prediction swings
with no new session data.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from f1pred.predict import _intra_weekend_history_row, _real_intra_weekend_rows


REF_DATE = datetime(2026, 7, 4, 14, 0, tzinfo=timezone.utc)


def _ranked_row(actual_position=np.nan, predicted_position=5, grid=3):
    return pd.Series({
        "driverId": "max_verstappen",
        "constructorId": "red_bull",
        "actual_position": actual_position,
        "predicted_position": predicted_position,
        "grid": grid,
    })


class TestIntraWeekendHistoryRow:
    def test_finished_session_uses_actual_and_is_not_predicted(self):
        row = _intra_weekend_history_row(
            _ranked_row(actual_position=2, predicted_position=5),
            "sprint_qualifying", REF_DATE,
        )
        assert row["position"] == 2
        assert row["predicted"] is False
        assert row["qpos"] == 2

    def test_unrun_session_falls_back_to_prediction_and_is_tagged(self):
        row = _intra_weekend_history_row(
            _ranked_row(actual_position=np.nan, predicted_position=5),
            "sprint", REF_DATE,
        )
        assert row["position"] == 5
        assert row["predicted"] is True

    def test_qpos_only_set_for_qualifying_sessions(self):
        race_row = _intra_weekend_history_row(_ranked_row(), "race", REF_DATE)
        assert np.isnan(race_row["qpos"])
        quali_row = _intra_weekend_history_row(_ranked_row(), "qualifying", REF_DATE)
        assert quali_row["qpos"] == 5

    def test_points_stay_nan_to_protect_team_form(self):
        row = _intra_weekend_history_row(_ranked_row(actual_position=1), "race", REF_DATE)
        assert np.isnan(row["points"])


class TestRealIntraWeekendRows:
    def test_predicted_rows_are_excluded_from_feature_history(self):
        accumulated = [
            _intra_weekend_history_row(
                _ranked_row(actual_position=3), "sprint_qualifying", REF_DATE),
            _intra_weekend_history_row(
                _ranked_row(actual_position=np.nan, predicted_position=1), "sprint", REF_DATE),
            _intra_weekend_history_row(
                _ranked_row(actual_position=np.nan, predicted_position=7), "qualifying", REF_DATE),
        ]
        real = _real_intra_weekend_rows(accumulated)
        assert [r["session"] for r in real] == ["sprint_qualifying"]

    def test_all_real_rows_pass_through(self):
        accumulated = [
            _intra_weekend_history_row(
                _ranked_row(actual_position=3), "sprint_qualifying", REF_DATE),
            _intra_weekend_history_row(
                _ranked_row(actual_position=4), "sprint", REF_DATE),
        ]
        assert len(_real_intra_weekend_rows(accumulated)) == 2

    def test_untagged_legacy_rows_are_treated_as_real(self):
        # Rows built elsewhere (without the predicted key) must not be dropped.
        legacy = [{"driverId": "alonso", "position": 6, "session": "sprint"}]
        assert _real_intra_weekend_rows(legacy) == legacy


class TestFeedbackAmplification:
    """Demonstrates the failure mode the filter prevents: a same-day
    predicted 'result' dominates the boosted form index."""

    def test_same_day_boosted_row_dominates_form_index(self):
        from f1pred.features import compute_form_indices

        # A season of real results: driver consistently finishes P10.
        real = pd.DataFrame({
            "driverId": ["d1"] * 8,
            "position": [10.0] * 8,
            "date": pd.to_datetime(
                [f"2026-{m:02d}-01" for m in range(3, 7) for _ in range(2)], utc=True),
            "session": ["race"] * 8,
            "points": [1.0] * 8,
            "season": [2026] * 8,
        })
        # One same-day fake "sprint result" claiming P1.
        fake = pd.DataFrame({
            "driverId": ["d1"],
            "position": [1.0],
            "date": [REF_DATE],
            "session": ["sprint"],
            "points": [np.nan],
            "season": [2026],
        })

        clean = compute_form_indices(
            real, ref_date=REF_DATE, half_life_days=120,
            current_season=2026, boost_factor=8.0, sprint_boost_factor=8.0)
        poisoned = compute_form_indices(
            pd.concat([real, fake], ignore_index=True), ref_date=REF_DATE,
            half_life_days=120, current_season=2026,
            boost_factor=8.0, sprint_boost_factor=8.0)

        clean_val = float(clean["form_index"].iloc[0])
        poisoned_val = float(poisoned["form_index"].iloc[0])
        # The single fake row moves the index by more than a full position —
        # this is the swing the extra-history filter must prevent.
        assert poisoned_val - clean_val > 1.0
