
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from f1pred.predict import run_predictions_for_event

def test_run_predictions_bypass_optimization(sample_roster):
    """Verify that run_predictions_for_event bypasses simulation when actual results are available."""

    # Setup mocks
    cfg = MagicMock()
    cfg.data_sources.fastf1.enabled = False
    cfg.app.model_version = "1.0.0"
    cfg.paths.cache_dir = "/tmp/f1pred_test_cache"

    jc = MagicMock()
    # Mock schedule/event resolution
    jc.get_season_schedule.return_value = [{"round": "1", "date": "2023-03-05", "time": "15:00:00Z", "raceName": "Bahrain GP"}]
    jc.get_latest_season_and_round.return_value = ("2023", "1")

    # Mock actual results available
    actual_results = {r["driverId"]: i+1 for i, r in enumerate(sample_roster.to_dict('records'))}

    with patch("f1pred.predict.JolpicaClient", return_value=jc), \
         patch("f1pred.predict.OpenMeteoClient"), \
         patch("f1pred.predict.PredictionCache"), \
         patch("f1pred.calibrate.CalibrationManager") as mock_cm, \
         patch("f1pred.features.build_roster", return_value=sample_roster) as mock_build_roster, \
         patch("f1pred.predict._get_actual_positions_for_session") as mock_get_actual, \
         patch("f1pred.features.build_session_features") as mock_build_features, \
         patch("f1pred.models.train_pace_model") as mock_train, \
         patch("f1pred.predict.print_session_console"):

        mock_cm_inst = mock_cm.return_value
        mock_cm_inst.load_weights.return_value = {}

        # Aligned Series for actual positions
        mock_get_actual.return_value = sample_roster["driverId"].map(actual_results)

        # Execute
        run_predictions_for_event(cfg, season="2023", rnd="1", sessions=["qualifying"])

        # Assertions
        mock_build_roster.assert_called()
        mock_get_actual.assert_called()

        # CRITICAL: build_session_features and train_pace_model should NOT be called
        mock_build_features.assert_not_called()
        mock_train.assert_not_called()

def test_run_predictions_no_bypass_when_no_results(sample_roster):
    """Verify that run_predictions_for_event proceeds to feature building when results are NOT available."""

    # Setup mocks
    cfg = MagicMock()
    cfg.data_sources.fastf1.enabled = False
    cfg.app.model_version = "1.0.0"
    cfg.paths.cache_dir = "/tmp/f1pred_test_cache"
    cfg.modelling.monte_carlo.draws = 10
    cfg.modelling.simulation.noise_factor = 0.1
    cfg.modelling.simulation.min_noise = 0.01
    cfg.modelling.simulation.max_penalty_base = 5.0

    jc = MagicMock()
    jc.get_season_schedule.return_value = [{"round": "1", "date": "2023-03-05", "time": "15:00:00Z", "raceName": "Bahrain GP"}]
    jc.get_latest_season_and_round.return_value = ("2023", "1")

    with patch("f1pred.predict.JolpicaClient", return_value=jc), \
         patch("f1pred.predict.OpenMeteoClient"), \
         patch("f1pred.predict.PredictionCache") as mock_cache_cls, \
         patch("f1pred.calibrate.CalibrationManager") as mock_cm, \
         patch("f1pred.features.build_roster", return_value=sample_roster) as mock_build_roster, \
         patch("f1pred.predict._get_actual_positions_for_session", return_value=None), \
         patch("f1pred.features.build_session_features") as mock_build_features, \
         patch("f1pred.models.train_pace_model") as mock_train, \
         patch("f1pred.features.collect_historical_results"), \
         patch("f1pred.simulate.simulate_grid") as mock_sim, \
         patch("f1pred.predict.print_session_console"):

        mock_cm_inst = mock_cm.return_value
        mock_cm_inst.load_weights.return_value = {}
        mock_cm_inst.check_calibration_needed.return_value = False

        mock_cache = mock_cache_cls.return_value
        mock_cache.get.return_value = None

        mock_build_features.return_value = (pd.DataFrame({'driverId': ['d1'], 'grid': [1]}), {"weather": {}}, sample_roster)
        mock_train.return_value = (MagicMock(), np.array([0.0]), ["feat1"])
        mock_sim.return_value = (np.zeros((1, 1)), np.array([1.0]), None)

        # Execute
        run_predictions_for_event(cfg, season="2023", rnd="1", sessions=["qualifying"])

        # Assertions
        mock_build_roster.assert_called()
        # Should proceed to heavy lifting
        mock_build_features.assert_called()
        mock_train.assert_called()
