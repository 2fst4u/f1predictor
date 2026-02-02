import unittest
import tempfile
import os
import yaml
from f1pred.config import load_config

class TestConfigSecurity(unittest.TestCase):
    def setUp(self):
        self.valid_config = {
            "app": {
                "model_version": "v1.0",
                "random_seed": 42,
                "timezone": "UTC",
                "live_refresh_seconds": 600,
                "log_level": "WARNING"
            },
            "paths": {
                "cache_dir": ".cache/test",
                "fastf1_cache": ".cache/test_ff1"
            },
            "data_sources": {
                "jolpica": {
                    "base_url": "https://api.jolpi.ca",
                    "timeout_seconds": 30,
                    "rate_limit_sleep": 0.5,
                    "enabled": True
                },
                "fastf1": {"enabled": False},
                "open_meteo": {
                    "forecast_url": "https://api.open-meteo.com",
                    "historical_weather_url": "https://archive-api.open-meteo.com",
                    "historical_forecast_url": "https://historical-forecast-api.open-meteo.com",
                    "elevation_url": "https://api.open-meteo.com",
                    "geocoding_url": "https://geocoding-api.open-meteo.com",
                    "enabled": True,
                    "temperature_unit": "celsius",
                    "windspeed_unit": "kmh",
                    "precipitation_unit": "mm"
                }
            },
            "caching": {
                "requests_cache": {
                    "backend": "sqlite",
                    "expire_after": {},
                    "allowable_codes": [200],
                    "stale_if_error": True
                }
            },
            "modelling": {
                "recency_half_life_days": {"base": 120, "weather": 180, "team": 240},
                "monte_carlo": {"draws": 5000},
                "features": {
                    "include_fastf1_fill": True,
                    "include_circuit_elevation": True,
                    "include_weather_ensemble": True
                },
                "targets": {"session_types": ["race"]},
                "ensemble": {
                    "w_elo": 0.2, "w_bt": 0.2, "w_mixed": 0.2, "w_gbm": 0.4, "min_std": 0.05
                },
                "simulation": {
                    "noise_factor": 0.15, "min_noise": 0.05, "max_penalty_base": 20.0
                },
                "blending": {
                    "gbm_weight": 0.75, "baseline_weight": 0.25,
                    "baseline_team_factor": 0.3, "baseline_driver_team_factor": 0.2
                },
                "dnf": {
                    "alpha": 2.0, "beta": 8.0, "driver_weight": 0.6,
                    "team_weight": 0.4, "clip_min": 0.02, "clip_max": 0.35
                }
            },
            "backtesting": {
                "enabled": False,
                "seasons": [],
                "metrics": []
            }
        }

    def _write_config(self, cfg_dict):
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, 'w') as f:
            yaml.dump(cfg_dict, f)
        return path

    def test_excessive_monte_carlo_draws(self):
        """
        Security Test: Ensure that excessive Monte Carlo draws are rejected
        to prevent Memory Exhaustion / DoS.
        """
        cfg = self.valid_config.copy()
        # Set draws to 1,000,001 (over likely limit of 100,000)
        cfg["modelling"]["monte_carlo"]["draws"] = 1000001

        path = self._write_config(cfg)
        try:
            with self.assertRaises(ValueError) as cm:
                load_config(path)
            self.assertIn("monte_carlo.draws", str(cm.exception))
        finally:
            os.remove(path)

    def test_insufficient_monte_carlo_draws(self):
        """
        Security Test: Ensure that too few Monte Carlo draws are rejected.
        """
        cfg = self.valid_config.copy()
        cfg["modelling"]["monte_carlo"]["draws"] = 10

        path = self._write_config(cfg)
        try:
            with self.assertRaises(ValueError) as cm:
                load_config(path)
            self.assertIn("monte_carlo.draws", str(cm.exception))
        finally:
            os.remove(path)

    def test_dangerous_refresh_rate(self):
        """
        Security Test: Ensure that extremely low refresh rates are rejected
        to prevent API abuse (DoS against upstream).
        """
        cfg = self.valid_config.copy()
        # Set refresh to 1 second
        cfg["app"]["live_refresh_seconds"] = 1

        path = self._write_config(cfg)
        try:
            with self.assertRaises(ValueError) as cm:
                load_config(path)
            self.assertIn("live_refresh_seconds", str(cm.exception))
        finally:
            os.remove(path)

if __name__ == "__main__":
    unittest.main()
