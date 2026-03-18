import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from f1pred.features import _cache_path_for_season, _load_season_cache, _save_season_cache

@pytest.fixture
def temp_cache_dir(tmp_path):
    path = tmp_path / "cache"
    path.mkdir()
    return str(path)

def test_cache_path_for_season():
    # Arrange & Act
    path = _cache_path_for_season("/tmp/cache", 2023)

    # Assert
    assert str(path).endswith("history/season_2023.parquet")

def test_save_and_load_season_cache(temp_cache_dir):
    # Arrange
    season = 2023
    df = pd.DataFrame({"driver_id": ["verstappen", "hamilton"], "points": [25, 18]})

    # Act: Save
    _save_season_cache(temp_cache_dir, season, df)

    # Assert: File exists
    path = Path(temp_cache_dir) / "history" / f"season_{season}.parquet"
    assert path.exists()

    # Act: Load
    loaded_df = _load_season_cache(temp_cache_dir, season)

    # Assert: Data matches
    assert loaded_df is not None
    pd.testing.assert_frame_equal(df, loaded_df)

def test_load_season_cache_not_exists(temp_cache_dir):
    # Arrange
    season = 2024

    # Act: Load non-existent
    loaded_df = _load_season_cache(temp_cache_dir, season)

    # Assert: None returned
    assert loaded_df is None

def test_save_season_cache_empty_df(temp_cache_dir):
    # Arrange
    season = 2025
    df = pd.DataFrame()

    # Act: Save empty
    _save_season_cache(temp_cache_dir, season, df)

    # Assert: File does not exist
    path = Path(temp_cache_dir) / "history" / f"season_{season}.parquet"
    assert not path.exists()

def test_load_season_cache_exception(temp_cache_dir):
    # Arrange
    season = 2026
    path = Path(temp_cache_dir) / "history" / f"season_{season}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("invalid parquet data") # Invalid file

    # Act: Load
    loaded_df = _load_season_cache(temp_cache_dir, season)

    # Assert: None returned due to exception
    assert loaded_df is None

@patch("f1pred.features.pd.DataFrame.to_parquet")
def test_save_season_cache_exception(mock_to_parquet, temp_cache_dir):
    # Arrange
    season = 2027
    df = pd.DataFrame({"col": [1]})

    # Setup mock to raise exception
    mock_to_parquet.side_effect = Exception("Mock exception")

    # Act: Save (should swallow exception and log)
    _save_season_cache(temp_cache_dir, season, df)

    # Assert: No file actually created by the mock
    path = Path(temp_cache_dir) / "history" / f"season_{season}.parquet"
    assert not path.exists()
