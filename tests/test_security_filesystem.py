
import os
import shutil
import pytest
import logging
from unittest.mock import patch
from pathlib import Path
from f1pred.util import ensure_dirs

@pytest.fixture
def temp_cache_dir(tmp_path):
    d = tmp_path / "cache_test"
    yield d
    if d.exists():
        shutil.rmtree(d)

def test_ensure_dirs_permissions(temp_cache_dir):
    """Verify that ensure_dirs creates directories with 0o700 permissions."""
    path_str = str(temp_cache_dir)
    ensure_dirs(path_str)

    assert temp_cache_dir.exists()

    stat = os.stat(path_str)
    mode = stat.st_mode & 0o777

    # Mode should be 0o700
    assert mode == 0o700, f"Expected 0o700, got {oct(mode)}"

def test_ensure_dirs_nested(temp_cache_dir):
    """Verify recursive creation and permission on leaf."""
    nested = temp_cache_dir / "subdir" / "leaf"
    path_str = str(nested)

    ensure_dirs(path_str)

    assert nested.exists()

    # Leaf should be 0o700
    stat = os.stat(path_str)
    mode = stat.st_mode & 0o777
    assert mode == 0o700, f"Expected 0o700 for leaf, got {oct(mode)}"

    # Parent (subdir) permissions are system dependent (mkdir default),
    # but that's acceptable as long as leaf is secure.
    # However, if ensure_dirs is called on parents explicitly, they should be secure.

def test_ensure_dirs_warning_on_failure(temp_cache_dir, caplog):
    """Verify that ensure_dirs logs a warning if chmod fails."""
    path_str = str(temp_cache_dir)

    # Mock Path.chmod to raise PermissionError
    with patch("pathlib.Path.chmod", side_effect=PermissionError("Mock error")):
        with caplog.at_level(logging.WARNING, logger="f1pred.util"):
            ensure_dirs(path_str)

    assert "Failed to set secure permissions" in caplog.text
    assert "Mock error" in caplog.text

def test_features_uses_ensure_dirs(temp_cache_dir):
    """Verify that feature caching functions utilize ensure_dirs."""
    from f1pred.features import _save_season_cache, _save_weather_cache
    import pandas as pd

    # Mock ensure_dirs in features.py
    with patch("f1pred.features.ensure_dirs") as mock_ensure:
        # 1. Season cache
        df = pd.DataFrame({"a": [1]})
        with patch("pandas.DataFrame.to_parquet"): # Don't actually write
            _save_season_cache(str(temp_cache_dir), 2025, df)

        assert mock_ensure.called
        # Check argument ends with 'history'
        args, _ = mock_ensure.call_args
        assert args[0].endswith("history")

        mock_ensure.reset_mock()

        # 2. Weather cache
        data = {"temp": 20.0}
        with patch("builtins.open"): # Don't open file
             _save_weather_cache(str(temp_cache_dir), 2025, 1, data)

        assert mock_ensure.called
        args, _ = mock_ensure.call_args
        assert args[0].endswith("weather")
