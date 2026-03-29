
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
    cfg.modelling.monte_carlo.draws = 2 # Speed up

    # Mock data for a session that has finished
    mock_roster = pd.DataFrame([
        {"driverId": "verstappen", "name": "Max Verstappen", "code": "VER", "constructorName": "Red Bull", "constructorId": "red_bull"},
        {"driverId": "hamilton", "name": "Lewis Hamilton", "code": "HAM", "constructorName": "Mercedes", "constructorId": "mercedes"},
    ])

    with patch("f1pred.predict.JolpicaClient") as MockJolpica, \
         patch("f1pred.predict.OpenMeteoClient") as MockMeteo, \
         patch("f1pred.predict.PredictionCache") as mock_cache_cls, \
         patch("f1pred.predict.build_roster") as mock_build_roster, \
         patch("f1pred.predict.build_session_features") as mock_build_features, \
         patch("f1pred.predict.get_session_classification") as mock_get_class, \
         patch("f1pred.predict._get_actual_positions_for_session") as mock_get_actual, \
         patch("f1pred.features.collect_historical_results") as mock_collect, \
         patch("f1pred.calibrate.CalibrationManager") as mock_cm_cls, \
         patch("f1pred.models.train_pace_model") as mock_train, \
         patch("f1pred.simulate.simulate_grid") as mock_sim, \
         patch("f1pred.predict.print_session_console"):

        jc = MockJolpica.return_value
        jc.get_season_schedule.return_value = [
            {"round": "1", "raceName": "Test GP", "date": "2020-04-01", "time": "12:00:00Z", "season": "2020"}
        ]
        jc.get_latest_season_and_round.return_value = ("2020", "1")

        mock_build_roster.return_value = mock_roster

        # Mock actuals Series
        mock_get_actual.return_value = pd.Series([2, 1], index=[0, 1]) # Verstappen 2, Hamilton 1

        mock_features = mock_roster.copy()
        mock_features["grid"] = [1, 2]
        mock_build_features.return_value = (mock_features, {"weather": {}}, mock_roster)

        mock_cache = mock_cache_cls.return_value
        mock_cache.get.return_value = None
        mock_cache.get_by_key.return_value = None

        mock_collect.return_value = pd.DataFrame()
        mock_cm = mock_cm_cls.return_value
        mock_cm.check_calibration_needed.return_value = False
        mock_cm.load_weights.return_value = {}

        # Mock model outputs
        mock_train.return_value = (MagicMock(), np.array([0.0, 0.1]), ["feat1"], None)
        # Hamilton (idx 1) has 80% win prob, Verstappen (idx 0) has 20%
        mock_sim.return_value = (np.zeros((2, 2)), np.array([0.2, 0.8]), None)

        # Run predictions
        results = run_predictions_for_event(
            cfg,
            season="2020",
            rnd="1",
            sessions=["race"],
            return_results=True,
            use_actuals=True
        )

        assert "race" in results["sessions"], f"Race session missing from results. Results: {results}"
        race_preds = results["sessions"]["race"]["ranked"]

        ham_row = race_preds[race_preds["driverId"] == "hamilton"].iloc[0]
        ver_row = race_preds[race_preds["driverId"] == "verstappen"].iloc[0]

        # Win probabilities should be from ML (0.8/0.2), NOT from actuals (1.0/0.0)
        assert ham_row["p_win"] == 0.8
        assert ver_row["p_win"] == 0.2

        # Actual positions should be correctly mapped
        assert ham_row["actual_position"] == 1
        assert ver_row["actual_position"] == 2
