
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from f1pred.predict import run_predictions_for_event
from f1pred.config import AppConfig

class TestGridLogic(unittest.TestCase):
    def setUp(self):
        self.cfg = MagicMock(spec=AppConfig)

        # Build nested mocks carefully
        self.cfg.paths = MagicMock()
        self.cfg.paths.cache_dir = "test_cache"
        self.cfg.paths.fastf1_cache = "test_fastf1_cache"

        self.cfg.data_sources = MagicMock()
        self.cfg.data_sources.jolpica = MagicMock()
        self.cfg.data_sources.jolpica.base_url = "http://localhost"
        self.cfg.data_sources.jolpica.timeout_seconds = 1
        self.cfg.data_sources.jolpica.rate_limit_sleep = 0
        self.cfg.data_sources.fastf1 = MagicMock()
        self.cfg.data_sources.fastf1.enabled = False

        self.cfg.app = MagicMock()
        self.cfg.app.model_version = "1.0"

        self.cfg.modelling = MagicMock()
        self.cfg.modelling.targets = MagicMock()
        self.cfg.modelling.targets.session_types = ["qualifying", "race"]
        self.cfg.modelling.monte_carlo = MagicMock()
        self.cfg.modelling.monte_carlo.draws = 10
        self.cfg.modelling.simulation = MagicMock()
        self.cfg.modelling.simulation.noise_factor = 0.1
        self.cfg.modelling.simulation.min_noise = 0.01
        self.cfg.modelling.simulation.max_penalty_base = 0.5

        self.cfg.caching = MagicMock()
        self.cfg.caching.prediction_cache = MagicMock()
        self.cfg.caching.prediction_cache.max_entries = 0

        self.cfg.modelling.recency_half_life_days = MagicMock()
        self.cfg.modelling.recency_half_life_days.base = 365
        self.cfg.modelling.recency_half_life_days.team = 730

        self.cfg.modelling.blending = MagicMock()
        self.cfg.modelling.blending.gbm_weight = 0.5
        self.cfg.modelling.blending.baseline_weight = 0.5

    @patch('f1pred.predict.JolpicaClient')
    @patch('f1pred.predict.OpenMeteoClient')
    @patch('f1pred.predict.resolve_event')
    @patch('f1pred.features.build_session_features')
    @patch('f1pred.predict._get_actual_positions_for_session')
    @patch('f1pred.models.train_pace_model')
    @patch('f1pred.simulate.simulate_grid')
    @patch('f1pred.calibrate.CalibrationManager')
    @patch('f1pred.features.collect_historical_results')
    @patch('f1pred.predict._run_single_prediction')
    def test_race_prediction_when_quali_has_results(self,
                                                     mock_run_single,
                                                     mock_collect_hist,
                                                     mock_cal_mgr,
                                                     mock_simulate,
                                                     mock_train,
                                                     mock_get_actuals,
                                                     mock_build_features,
                                                     mock_resolve,
                                                     mock_om,
                                                     mock_jc):
        # Setup mocks
        mock_resolve.return_value = (2024, 1, {"raceName": "Bahrain GP", "date": "2024-03-02", "time": "15:00:00Z"})

        roster = pd.DataFrame({
            "driverId": ["max_verstappen", "sergio_perez"],
            "constructorId": ["red_bull", "red_bull"],
            "number": ["1", "11"],
            "code": ["VER", "PER"],
            "name": ["Max Verstappen", "Sergio Perez"]
        })

        # X for race, initially missing grid for one driver
        X_race = roster.copy()
        X_race["grid"] = [1.0, np.nan] # Perez missing grid

        mock_build_features.return_value = (X_race, {"weather": {}}, roster)

        # Actuals for race: None
        # Actuals for qualifying: available
        def side_effect_get_actuals(jc, season, round, sess, roster_view):
            if sess == "race":
                return None
            if sess == "qualifying":
                return pd.Series([1, 2], index=[0, 1])
            return None
        mock_get_actuals.side_effect = side_effect_get_actuals

        mock_train.return_value = (None, np.array([0.0, 0.1]), [])
        mock_simulate.return_value = (np.eye(2), np.array([1.0, 2.0]), None)

        mock_cal_mgr.return_value.load_weights.return_value = {}
        mock_cal_mgr.return_value.check_calibration_needed.return_value = False

        # Run prediction for ONLY race
        run_predictions_for_event(self.cfg, "2024", "1", ["race"])

        # Verification
        # Since X_race["grid"] had a NaN, it should have triggered grid estimation logic.
        # It SHOULD HAVE checked for actual results of qualifying first,
        # and since we mocked them to be available, it should NOT call _run_single_prediction for qualifying.

        # _run_single_prediction should not be called at all in this scenario
        mock_run_single.assert_not_called()

    @patch('f1pred.predict.JolpicaClient')
    @patch('f1pred.predict.OpenMeteoClient')
    @patch('f1pred.predict.resolve_event')
    @patch('f1pred.features.build_session_features')
    @patch('f1pred.predict._get_actual_positions_for_session')
    @patch('f1pred.models.train_pace_model')
    @patch('f1pred.simulate.simulate_grid')
    @patch('f1pred.calibrate.CalibrationManager')
    @patch('f1pred.features.collect_historical_results')
    @patch('f1pred.predict._run_single_prediction')
    def test_race_prediction_when_no_actuals_forces_simulation(self,
                                                                mock_run_single,
                                                                mock_collect_hist,
                                                                mock_cal_mgr,
                                                                mock_simulate,
                                                                mock_train,
                                                                mock_get_actuals,
                                                                mock_build_features,
                                                                mock_resolve,
                                                                mock_om,
                                                                mock_jc):
        # Setup mocks
        mock_resolve.return_value = (2024, 1, {"raceName": "Bahrain GP", "date": "2024-03-02", "time": "15:00:00Z"})

        roster = pd.DataFrame({
            "driverId": ["max_verstappen", "sergio_perez"],
            "constructorId": ["red_bull", "red_bull"],
            "number": ["1", "11"],
            "code": ["VER", "PER"],
            "name": ["Max Verstappen", "Sergio Perez"]
        })

        # X for race, missing grid for Perez
        X_race = roster.copy()
        X_race["grid"] = [1.0, np.nan]

        mock_build_features.return_value = (X_race, {"weather": {}}, roster)

        # No actuals at all
        mock_get_actuals.return_value = None

        # Mock simulation result for qualifying
        mock_run_single.return_value = pd.DataFrame({
            "driverId": ["max_verstappen", "sergio_perez"],
            "predicted_position": [1, 2]
        })

        mock_train.return_value = (None, np.array([0.0, 0.1]), [])
        mock_simulate.return_value = (np.eye(2), np.array([1.0, 2.0]), None)

        mock_cal_mgr.return_value.load_weights.return_value = {}
        mock_cal_mgr.return_value.check_calibration_needed.return_value = False

        # Run prediction for ONLY race
        run_predictions_for_event(self.cfg, "2024", "1", ["race"])

        # Verification: _run_single_prediction should be called for qualifying
        mock_run_single.assert_called_once()
        args, _ = mock_run_single.call_args
        self.assertEqual(args[4], "qualifying")

        # Verify that X["grid"] was updated correctly (the one passed to train_pace_model)
        called_X = mock_train.call_args[0][0]
        self.assertEqual(called_X.loc[1, "grid"], 2)

if __name__ == "__main__":
    unittest.main()
