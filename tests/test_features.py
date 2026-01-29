"""Tests for feature engineering."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import os
from f1pred.features import (
    exponential_weights,
    compute_form_indices,
    compute_weather_sensitivity,
    _fetch_weather_task,
    _WEATHER_EVENT_CACHE,
)
from f1pred.data.open_meteo import OpenMeteoClient


def test_exponential_weights():
    """Test exponential weight calculation."""
    ref_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    dates = [
        ref_date - timedelta(days=365),
        ref_date - timedelta(days=180),
        ref_date - timedelta(days=30),
        ref_date,
    ]
    weights = exponential_weights(dates, ref_date, half_life_days=365)

    assert len(weights) == 4
    # Most recent should have highest weight
    assert weights[-1] == 1.0
    # Older dates should have lower weights
    assert weights[0] < weights[-1]
    assert np.all(weights > 0)


def test_compute_form_indices(sample_historical_data):
    """Test form index calculation."""
    ref_date = datetime(2023, 3, 10, tzinfo=timezone.utc)
    form_df = compute_form_indices(sample_historical_data, ref_date, half_life_days=365)

    assert 'driverId' in form_df.columns
    assert 'form_index' in form_df.columns
    assert len(form_df) > 0
    # Form indices should be numeric and finite
    assert np.all(np.isfinite(form_df['form_index']))


# Note: teammate_head_to_head was removed in a refactor; test removed


@pytest.fixture
def mock_om():
    om = MagicMock(spec=OpenMeteoClient)
    return om


@pytest.fixture
def temp_cache_dir(tmp_path):
    path = tmp_path / "cache"
    path.mkdir()
    return str(path)


def test_fetch_weather_task_caching(mock_om, temp_cache_dir):
    # Setup
    lat, lon = 10.0, 20.0
    dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
    season, rnd = 2023, 1

    # Mock aggregation
    expected_agg = {"temp_mean": 25.0}
    with patch("f1pred.features._aggregate_weather", return_value=expected_agg) as mock_agg:

        # 1. First call: Should hit API and save to cache
        res1 = _fetch_weather_task(mock_om, lat, lon, dt, season, rnd, temp_cache_dir)
        assert res1 == expected_agg
        mock_agg.assert_called_once()

        # Verify file created
        weather_file = os.path.join(temp_cache_dir, "weather", f"event_{season}_{rnd}.json")
        assert os.path.exists(weather_file)

        # 2. Second call: Should hit disk cache, NOT API
        mock_agg.reset_mock()
        res2 = _fetch_weather_task(mock_om, lat, lon, dt, season, rnd, temp_cache_dir)
        assert res2 == expected_agg
        mock_agg.assert_not_called()


def test_compute_weather_sensitivity_uses_cache(mock_om, temp_cache_dir):
    # Setup Data
    hist = pd.DataFrame({
        "season": [2023, 2023],
        "round": [1, 2],
        "circuit": ["A", "B"],
        "lat": [10.0, 11.0],
        "lon": [20.0, 21.0],
        "date": [datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 1, 8, tzinfo=timezone.utc)],
        "session": ["race", "race"],
        "driverId": ["d1", "d1"],
        "position": [1, 2]
    })
    roster = pd.DataFrame({"driverId": ["d1"]})
    ref_date = datetime(2023, 2, 1, tzinfo=timezone.utc)

    # Reset in-memory cache
    _WEATHER_EVENT_CACHE.clear()

    expected_agg = {"temp_mean": 25.0}

    with patch("f1pred.features._aggregate_weather", return_value=expected_agg) as mock_agg:
        # 1. First run: 2 API calls
        compute_weather_sensitivity(mock_om, hist, roster, ref_date, cache_dir=temp_cache_dir)
        assert mock_agg.call_count == 2

        # 2. Second run: 0 API calls (should use disk cache via _fetch_weather_task or in-memory)
        # Clear in-memory cache to force disk check, OR rely on in-memory check
        # Let's test disk cache by clearing in-memory
        _WEATHER_EVENT_CACHE.clear()
        mock_agg.reset_mock()

        compute_weather_sensitivity(mock_om, hist, roster, ref_date, cache_dir=temp_cache_dir)
        assert mock_agg.call_count == 0
