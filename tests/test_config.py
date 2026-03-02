"""Tests for config loading — happy path and validation."""
import pytest
import os
import tempfile
import yaml
from f1pred.config import load_config, AppConfig


@pytest.fixture
def full_valid_config():
    """A complete, valid config dict that should load without errors."""
    return {
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
        },
        "calibration": {
            "enabled": False,
            "lookback_window_days": 365,
            "frequency_hours": 24,
            "weights_file": "calibration_weights.json"
        }
    }


def _write_config(cfg_dict):
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, 'w') as f:
        yaml.dump(cfg_dict, f)
    return path


def test_load_config_happy_path(full_valid_config):
    """Verify that a complete valid config loads into AppConfig correctly."""
    path = _write_config(full_valid_config)
    try:
        cfg = load_config(path)
        assert isinstance(cfg, AppConfig)
        assert cfg.app.model_version == "v1.0"
        assert cfg.app.random_seed == 42
        assert cfg.modelling.monte_carlo.draws == 5000
        assert cfg.modelling.ensemble.w_gbm == 0.4
        assert cfg.calibration.enabled is False
    finally:
        os.remove(path)


def test_load_config_missing_section(full_valid_config):
    """Verify that a missing top-level section raises ValueError."""
    del full_valid_config["calibration"]
    path = _write_config(full_valid_config)
    try:
        with pytest.raises(ValueError, match="Missing top-level section"):
            load_config(path)
    finally:
        os.remove(path)


def test_load_config_boundary_draws(full_valid_config):
    """Test boundary values for monte_carlo.draws validation."""
    # Exactly 100 — should be valid (minimum boundary)
    full_valid_config["modelling"]["monte_carlo"]["draws"] = 100
    path = _write_config(full_valid_config)
    try:
        cfg = load_config(path)
        assert cfg.modelling.monte_carlo.draws == 100
    finally:
        os.remove(path)

    # Exactly 100_000 — should be valid (maximum boundary)
    full_valid_config["modelling"]["monte_carlo"]["draws"] = 100_000
    path = _write_config(full_valid_config)
    try:
        cfg = load_config(path)
        assert cfg.modelling.monte_carlo.draws == 100_000
    finally:
        os.remove(path)


def test_load_config_invalid_session_type(full_valid_config):
    """Verify that invalid session types are rejected."""
    full_valid_config["modelling"]["targets"]["session_types"] = ["race", "banana"]
    path = _write_config(full_valid_config)
    try:
        with pytest.raises(ValueError, match="unknown entries"):
            load_config(path)
    finally:
        os.remove(path)

def test_load_config_missing_paths(full_valid_config):
    """Verify that missing paths raise ValueError."""
    del full_valid_config["paths"]["cache_dir"]
    path = _write_config(full_valid_config)
    try:
        with pytest.raises(ValueError, match="Missing paths.cache_dir"):
            load_config(path)
    finally:
        os.remove(path)

def test_load_config_invalid_urls(full_valid_config):
    """Verify that non-http URLs are rejected."""
    full_valid_config["data_sources"]["jolpica"]["base_url"] = "ftp://api.jolpi.ca"
    full_valid_config["data_sources"]["open_meteo"]["forecast_url"] = "ftp://api.open-meteo.com"
    path = _write_config(full_valid_config)
    try:
        with pytest.raises(ValueError) as excinfo:
            load_config(path)
        err_msg = str(excinfo.value)
        assert "data_sources.jolpica.base_url must be http(s) URL" in err_msg
        assert "data_sources.open_meteo.forecast_url must be http(s) URL" in err_msg
    finally:
        os.remove(path)

def test_load_config_invalid_units(full_valid_config):
    """Verify that invalid weather units are rejected."""
    full_valid_config["data_sources"]["open_meteo"]["temperature_unit"] = "kelvin"
    full_valid_config["data_sources"]["open_meteo"]["windspeed_unit"] = "knots"
    full_valid_config["data_sources"]["open_meteo"]["precipitation_unit"] = "liters"
    path = _write_config(full_valid_config)
    try:
        with pytest.raises(ValueError) as excinfo:
            load_config(path)
        err_msg = str(excinfo.value)
        assert "data_sources.open_meteo.temperature_unit must be one of" in err_msg
        assert "data_sources.open_meteo.windspeed_unit must be one of" in err_msg
        assert "data_sources.open_meteo.precipitation_unit must be one of" in err_msg
    finally:
        os.remove(path)
