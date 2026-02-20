import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from pathlib import Path
from f1pred.calibrate import CalibrationManager

class TestCalibrationTrigger(unittest.TestCase):
    def setUp(self):
        self.mock_cfg = MagicMock()
        self.mock_cfg.calibration.enabled = True
        self.mock_cfg.calibration.weights_file = Path("test_weights.json")
        self.mock_cfg.calibration.lookback_window_days = 365
        self.mock_cfg.calibration.frequency_hours = 24

    def test_calibration_check_missing_file(self):
        # Mock Path.exists to return False
        with patch.object(Path, "exists", return_value=False):
            cm = CalibrationManager(self.mock_cfg)
            # Override weights_file to be a Path object that returns False on exists()
            # Actually easiest is just to mock the instance attribute if needed, 
            # but CalibrationManager uses self.weights_file.exists()
            # The patch above mocks ALL Path.exists calls which might be dangerous?
            # Better to mock the property on the instance.
            cm.weights_file = MagicMock()
            cm.weights_file.exists.return_value = False
            
            self.assertTrue(cm.check_calibration_needed())

    def test_calibration_check_new_race(self):
        cm = CalibrationManager(self.mock_cfg)
        cm.weights_file = MagicMock()
        cm.weights_file.exists.return_value = True
        
        # We need to ensure load_weights DOESN'T overwrite our manual set of last_race_id
        # In reality, load_weights reads from disk. We mock it to do nothing.
        with patch.object(cm, "load_weights", return_value={}):
            cm.last_race_id = "2024_20"
            
            # New history with a newer race (ID will be 2024_21)
            history_df = pd.DataFrame([
                {"session": "race", "date": pd.Timestamp("2024-12-01"), "season": 2024, "round": 21, "position": 1}
            ])
            
            # Should return True because 2024_21 != 2024_20
            self.assertTrue(cm.check_calibration_needed(history_df))

    def test_calibration_check_same_race(self):
        cm = CalibrationManager(self.mock_cfg)
        cm.weights_file = MagicMock()
        cm.weights_file.exists.return_value = True
        
        with patch.object(cm, "load_weights", return_value={}):
            cm.last_race_id = "2024_21"
            
            # History with same latest race
            history_df = pd.DataFrame([
                {"session": "race", "date": pd.Timestamp("2024-12-01"), "season": 2024, "round": 21, "position": 1}
            ])
            
            # Should return False because IDs match
            self.assertFalse(cm.check_calibration_needed(history_df))

if __name__ == "__main__":
    unittest.main()
