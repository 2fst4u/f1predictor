
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from f1pred.predict import run_predictions_for_event
from f1pred.config import load_config

def test_finished_session_retains_ml_stats():
    """
    Verify that even when actual results are available, the ML prediction
    pipeline still runs and provides probabilistic stats, instead of
    just cloning the actual results into the prediction fields.
    """
    cfg = load_config("config.yaml")

    # Mock data for a session that has finished
    mock_roster = pd.DataFrame([
        {"driverId": "verstappen", "name": "Max Verstappen", "code": "VER", "constructorName": "Red Bull", "constructorId": "red_bull"},
        {"driverId": "hamilton", "name": "Lewis Hamilton", "code": "HAM", "constructorName": "Mercedes", "constructorId": "mercedes"},
    ])

    # Actual results: Hamilton won, Verstappen 2nd
    mock_actual_results = [
        {"Driver": {"driverId": "hamilton"}, "position": "1"},
        {"Driver": {"driverId": "verstappen"}, "position": "2"},
    ]

    with patch("f1pred.predict.JolpicaClient") as MockJolpica, \
         patch("f1pred.predict.OpenMeteoClient") as MockMeteo, \
         patch("f1pred.features.build_roster") as mock_build_roster, \
         patch("f1pred.features.build_session_features") as mock_build_features, \
         patch("f1pred.predict.get_session_classification") as mock_get_class:

        jc = MockJolpica.return_value
        jc.get_season_schedule.return_value = [
            {"round": "1", "raceName": "Test GP", "date": "2026-04-01", "time": "12:00:00Z", "season": "2026"}
        ]
        # Jolpica returns Hamilton P1, Verstappen P2
        jc.get_race_results.return_value = mock_actual_results

        mock_build_roster.return_value = mock_roster

        # Mock features - just enough to run the model
        mock_features = mock_roster.copy()
        mock_features["grid"] = [1, 2]
        mock_features["form_index"] = [0.9, 0.8]

        mock_build_features.return_value = (mock_features, {"weather": {}}, mock_roster)

        # Run predictions
        results = run_predictions_for_event(
            cfg,
            season="2026",
            rnd="1",
            sessions=["race"],
            return_results=True,
            use_actuals=True # This is the default and the one we want to test
        )

        race_preds = results["sessions"]["race"]["ranked"]

        # If the bug is present, race_preds will be sorted by actual position (Hamilton 1st)
        # and predicted_position will be equal to actual_position.
        # Win probabilities will be 1.0 for Hamilton and 0.0 for Verstappen.

        ham_row = race_preds[race_preds["driverId"] == "hamilton"].iloc[0]
        ver_row = race_preds[race_preds["driverId"] == "verstappen"].iloc[0]

        print(f"Hamilton Predicted Pos: {ham_row['predicted_position']}, Actual: {ham_row['actual_position']}, Win Prob: {ham_row['p_win']}")
        print(f"Verstappen Predicted Pos: {ver_row['predicted_position']}, Actual: {ver_row['actual_position']}, Win Prob: {ver_row['p_win']}")

        # We want ML stats to be probabilistic, not 1.0/0.0
        assert 0.0 < ham_row["p_win"] < 1.0
        assert 0.0 < ver_row["p_win"] < 1.0

        # We expect the prediction model to still exist.
        assert "shap_values" in race_preds.columns
        assert race_preds["shap_values"].notna().any()
