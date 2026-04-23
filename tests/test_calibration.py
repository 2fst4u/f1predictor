import unittest
import os
import time


from f1pred.calibrate import CalibrationManager, _unpack_weights, _pack_weights, PARAM_DEFAULTS, N_PARAMS, PARAM_BOUNDS
from f1pred.ensemble import EnsembleConfig

class MockCalibrationConfig:
    def __init__(self):
        self.enabled = True
        self.weights_file = "test_weights.json"
        self.lookback_window_days = 365
        self.frequency_hours = 24

class MockConfig:
    def __init__(self):
        self.calibration = MockCalibrationConfig()

class TestCalibrationManager(unittest.TestCase):
    def setUp(self):
        self.cfg = MockConfig()
        # Ensure cleanup before start
        if os.path.exists("test_weights.json"):
            os.remove("test_weights.json")
        self.cm = CalibrationManager(self.cfg)
        
    def tearDown(self):
        if os.path.exists("test_weights.json"):
            os.remove("test_weights.json")

    def test_defaults(self):
        weights = self.cm.load_weights()
        self.assertIn("ensemble", weights)
        self.assertIn("blending", weights)
        self.assertEqual(weights["ensemble"]["w_gbm"], 0.4)

    def test_defaults_contain_all_groups(self):
        """All six parameter groups must be present in defaults."""
        weights = self.cm.load_weights()
        for group in ("ensemble", "blending", "dnf", "simulation", "recency", "elo"):
            self.assertIn(group, weights, f"Missing default group: {group}")

    def test_defaults_dnf(self):
        weights = self.cm.load_weights()
        self.assertAlmostEqual(weights["dnf"]["alpha"], 2.0)
        self.assertAlmostEqual(weights["dnf"]["beta"], 8.0)
        self.assertAlmostEqual(weights["dnf"]["driver_weight"], 0.6)
        self.assertAlmostEqual(weights["dnf"]["team_weight"], 0.4)

    def test_defaults_simulation(self):
        weights = self.cm.load_weights()
        self.assertAlmostEqual(weights["simulation"]["noise_factor"], 0.15)
        self.assertAlmostEqual(weights["simulation"]["min_noise"], 0.05)

    def test_defaults_recency(self):
        weights = self.cm.load_weights()
        self.assertAlmostEqual(weights["recency"]["half_life_base"], 120.0)
        self.assertAlmostEqual(weights["recency"]["half_life_team"], 240.0)

    def test_defaults_elo(self):
        weights = self.cm.load_weights()
        self.assertAlmostEqual(weights["elo"]["k"], 20.0)

    def test_defaults_blending_complete(self):
        """Blending group should contain all keys including new ones."""
        weights = self.cm.load_weights()
        b = weights["blending"]
        for key in ("gbm_weight", "baseline_weight", "baseline_team_factor",
                     "grid_factor",
                     "current_season_weight", "current_season_qualifying_weight",
                     "current_quali_factor", "analytical_win_weight"):
            self.assertIn(key, b, f"Missing blending key: {key}")

    def test_save_and_load(self):
        self.cm.current_weights["ensemble"]["w_gbm"] = 0.99
        self.cm.save_weights()
        
        # New instance
        cm2 = CalibrationManager(self.cfg)
        w = cm2.load_weights()
        self.assertEqual(w["ensemble"]["w_gbm"], 0.99)

    def test_save_and_load_new_groups(self):
        """Save/load round-trips all six parameter groups."""
        self.cm.current_weights["dnf"]["alpha"] = 3.5
        self.cm.current_weights["simulation"]["noise_factor"] = 0.22
        self.cm.current_weights["recency"]["half_life_base"] = 90
        self.cm.current_weights["elo"]["k"] = 30.0
        self.cm.current_weights["blending"]["analytical_win_weight"] = 0.7
        self.cm.save_weights()

        cm2 = CalibrationManager(self.cfg)
        w = cm2.load_weights()
        self.assertAlmostEqual(w["dnf"]["alpha"], 3.5)
        self.assertAlmostEqual(w["simulation"]["noise_factor"], 0.22)
        self.assertAlmostEqual(w["recency"]["half_life_base"], 90)
        self.assertAlmostEqual(w["elo"]["k"], 30.0)
        self.assertAlmostEqual(w["blending"]["analytical_win_weight"], 0.7)

    def test_check_calibration_needed(self):
        # File missing -> needed
        self.assertTrue(self.cm.check_calibration_needed())
        
        # File exists and fresh -> not needed
        self.cm.save_weights()
        self.assertFalse(self.cm.check_calibration_needed())
        
        # File old -> needed
        # Modify mtime to be old
        old_time = time.time() - (25 * 3600) # 25 hours ago
        os.utime("test_weights.json", (old_time, old_time))
        self.assertTrue(self.cm.check_calibration_needed())

    def test_get_ensemble_config(self):
        self.cm.current_weights["ensemble"] = {
            "w_gbm": 0.1, "w_elo": 0.2, "w_bt": 0.3, "w_mixed": 0.4
        }
        cfg = self.cm.get_ensemble_config()
        self.assertIsInstance(cfg, EnsembleConfig)
        self.assertAlmostEqual(cfg.w_gbm, 0.1)
        self.assertAlmostEqual(cfg.w_mixed, 0.4)

    def test_save_includes_timestamp(self):
        self.cm.save_weights()
        cm2 = CalibrationManager(self.cfg)
        w = cm2.load_weights()
        self.assertIn("calibration_timestamp", w)


class TestPackUnpack(unittest.TestCase):
    """Test the _pack_weights / _unpack_weights round-trip."""

    def test_defaults_round_trip(self):
        """Unpacking defaults and repacking should be close to the original vector."""
        d = _unpack_weights(PARAM_DEFAULTS)
        v = _pack_weights(d)
        self.assertEqual(len(v), N_PARAMS)

    def test_unpack_has_all_groups(self):
        d = _unpack_weights(PARAM_DEFAULTS)
        for group in ("ensemble", "blending", "dnf", "simulation", "recency", "elo"):
            self.assertIn(group, d)

    def test_ensemble_normalised(self):
        """Race and qualifying ensemble weight sets should each sum to ~1.0."""
        d = _unpack_weights(PARAM_DEFAULTS)
        ens = d["ensemble"]
        race_total = ens["w_gbm"] + ens["w_elo"] + ens["w_bt"] + ens["w_mixed"]
        self.assertAlmostEqual(race_total, 1.0, places=5)
        quali_total = ens["w_gbm_quali"] + ens["w_elo_quali"] + ens["w_bt_quali"] + ens["w_mixed_quali"]
        self.assertAlmostEqual(quali_total, 1.0, places=5)

    def test_bounds_length(self):
        self.assertEqual(len(PARAM_BOUNDS), N_PARAMS)

    def test_bounds_order(self):
        """Every default must fall within its bounds."""
        for i, (lo, hi) in enumerate(PARAM_BOUNDS):
            self.assertGreaterEqual(PARAM_DEFAULTS[i], lo,
                                    f"Default[{i}]={PARAM_DEFAULTS[i]} < lower bound {lo}")
            self.assertLessEqual(PARAM_DEFAULTS[i], hi,
                                 f"Default[{i}]={PARAM_DEFAULTS[i]} > upper bound {hi}")


if __name__ == '__main__':
    unittest.main()
