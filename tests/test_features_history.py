
import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import MagicMock
from f1pred.features import collect_historical_results
from f1pred.data.jolpica import JolpicaClient

def test_collect_historical_results_empty():
    """Test handling of empty or failed API results."""
    jc = MagicMock(spec=JolpicaClient)
    jc.get_season_race_results.return_value = []
    jc.get_season_qualifying_results.return_value = []
    jc.get_season_sprint_results.return_value = []

    season = 2023
    end_before = datetime(2023, 1, 1, tzinfo=timezone.utc)

    df = collect_historical_results(jc, season, end_before, lookback_years=1)

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    # Should have at least the columns we define as default fallback
    assert "driverId" in df.columns

def test_collect_historical_results_data():
    """Test data collection and dataframe construction."""
    jc = MagicMock(spec=JolpicaClient)

    # Mock return data
    mock_race = [{
        "season": "2023", "round": "1", "date": "2023-03-05", "time": "15:00:00Z",
        "Circuit": {"circuitName": "Bahrain", "Location": {"lat": "26.0", "long": "50.5"}},
        "Results": [{
            "Driver": {"driverId": "max_verstappen", "code": "VER"},
            "Constructor": {"constructorId": "red_bull"},
            "grid": "1", "position": "1", "points": "25", "status": "Finished"
        }]
    }]

    def side_effect(yr):
        if str(yr) == "2023":
            return mock_race
        return []

    jc.get_season_race_results.side_effect = side_effect
    jc.get_season_qualifying_results.return_value = []
    jc.get_season_sprint_results.return_value = []

    season = 2023
    end_before = datetime(2023, 12, 31, tzinfo=timezone.utc) # End of year

    df = collect_historical_results(jc, season, end_before, lookback_years=1)

    assert not df.empty
    assert len(df) == 1
    assert df.iloc[0]["driverId"] == "max_verstappen"
    assert df.iloc[0]["points"] == 25.0
    assert df.iloc[0]["season"] == 2023
    assert df.iloc[0]["lat"] == 26.0

def test_collect_historical_results_date_filter():
    """Test that results after cutoff are filtered out."""
    jc = MagicMock(spec=JolpicaClient)

    mock_race = [{
        "season": "2023", "round": "1", "date": "2023-03-05", "time": "15:00:00Z",
        "Results": [{"Driver": {"driverId": "d1"}}]
    }]

    jc.get_season_race_results.return_value = mock_race
    jc.get_season_qualifying_results.return_value = []
    jc.get_season_sprint_results.return_value = []

    # Cutoff BEFORE the race
    season = 2023
    end_before = datetime(2023, 2, 1, tzinfo=timezone.utc)

    df = collect_historical_results(jc, season, end_before, lookback_years=1)

    assert df.empty
