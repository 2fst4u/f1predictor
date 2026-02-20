import unittest
from unittest.mock import MagicMock, patch
import json
import os
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import shutil

# Ensure path is correct relative to module import if needed, or import from package
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1pred.calibrate import CalibrationManager
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

    def test_save_and_load(self):
        self.cm.current_weights["ensemble"]["w_gbm"] = 0.99
        self.cm.save_weights()
        
        # New instance
        cm2 = CalibrationManager(self.cfg)
        w = cm2.load_weights()
        self.assertEqual(w["ensemble"]["w_gbm"], 0.99)

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

if __name__ == '__main__':
    unittest.main()
