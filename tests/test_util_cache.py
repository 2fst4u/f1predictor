import os
import logging
from datetime import datetime
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pandas as pd

from f1pred.util import PredictionCache

def test_cache_rolling_deletion(tmp_path):
    # Set up cache with max_entries = 2
    cache = PredictionCache(cache_dir=str(tmp_path), max_entries=2)

    # Generate 3 keys and save them, forcing mtime to ensure order
    base_time = 1000000000.0
    for i in range(3):
        cache.set({"key": i}, {"data": i})
        # Find the newly created file and update its mtime
        key = cache._generate_key({"key": i})
        cache_file = cache.cache_dir / f"{key}.json"
        os.utime(cache_file, (base_time + i, base_time + i))

    # Check that only the last 2 are kept
    files = list(cache.cache_dir.glob("*.json"))
    assert len(files) == 2

def test_cache_serialization_deserialization(tmp_path):
    cache = PredictionCache(cache_dir=str(tmp_path), max_entries=2)
    inputs = {"test": 1}
    df = pd.DataFrame({"a": [1, 2]})
    results = {
        "ranked": df,
        "prob_matrix": np.array([[0.5, 0.5]]),
        "pairwise": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "extra": "value"
    }

    cache.set(inputs, results)

    data = cache.get(inputs)
    assert data is not None
    assert isinstance(data["ranked"], pd.DataFrame)
    assert data["ranked"].equals(df)
    assert np.array_equal(data["prob_matrix"], results["prob_matrix"])
    assert np.array_equal(data["pairwise"], results["pairwise"])
    assert data["extra"] == "value"

def test_cache_get_miss(tmp_path):
    cache = PredictionCache(cache_dir=str(tmp_path), max_entries=2)
    inputs = {"test": "miss"}
    data = cache.get(inputs)
    assert data is None

@dataclass
class DummyDataclass:
    a: int
    b: str

def test_cache_extended_serialization(tmp_path):
    cache = PredictionCache(cache_dir=str(tmp_path), max_entries=2)
    dt = datetime(2023, 1, 1, 12, 0)
    ts = pd.Timestamp("2024-01-01")
    dc = DummyDataclass(a=1, b="test")

    inputs = {
        "dt": dt,
        "ts": ts,
        "dc": dc,
        "lst": [1, 2, dt]
    }
    results = {"val": 42}

    cache.set(inputs, results)
    data = cache.get(inputs)
    assert data is not None
    assert data["val"] == 42

def test_cache_get_error_handling(tmp_path, caplog):
    cache = PredictionCache(cache_dir=str(tmp_path), max_entries=2)
    inputs = {"test": 1}
    results = {"val": 42}
    cache.set(inputs, results)

    key = cache._generate_key(inputs)
    cache_file = cache.cache_dir / f"{key}.json"

    # Corrupt the JSON file to trigger Exception during load
    with open(cache_file, "w") as f:
        f.write("{invalid_json:")

    with caplog.at_level(logging.WARNING):
        data = cache.get(inputs)

    assert data is None
    assert "Failed to load prediction cache" in caplog.text

def test_cache_set_error_handling(tmp_path, caplog):
    cache = PredictionCache(cache_dir=str(tmp_path), max_entries=2)

    inputs = {"test": 2}
    results = {"val": 42}

    with patch.object(cache, '_generate_key', return_value="nonexistent_dir/file"):
        with caplog.at_level(logging.WARNING):
            cache.set(inputs, results)
        assert "Failed to save prediction cache" in caplog.text

def test_cache_rolling_deletion_unlink_fails(tmp_path):
    cache = PredictionCache(cache_dir=str(tmp_path), max_entries=2)
    for i in range(2):
        cache.set({"k": i}, {"v": i})

    # Cause unlink to raise an exception by mocking pathlib.Path.unlink
    with patch("pathlib.Path.unlink") as mock_unlink:
        mock_unlink.side_effect = OSError("Mock unlink error")

        # Trigger deletion
        cache.set({"k": 3}, {"v": 3})

    # Assert that exception was caught and ignored (files won't actually be deleted)
    assert len(list(cache.cache_dir.glob("*.json"))) == 3

def test_cache_get_by_key_set_by_key_success(tmp_path):
    cache = PredictionCache(cache_dir=str(tmp_path), max_entries=2)
    key = "explicit_key_123"
    df = pd.DataFrame({"a": [1, 2]})
    results = {
        "ranked": df,
        "prob_matrix": np.array([[0.5, 0.5]]),
        "pairwise": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "extra": "value"
    }

    cache.set_by_key(key, results)

    data = cache.get_by_key(key)
    assert data is not None
    assert isinstance(data["ranked"], pd.DataFrame)
    assert data["ranked"].equals(df)
    assert np.array_equal(data["prob_matrix"], results["prob_matrix"])
    assert np.array_equal(data["pairwise"], results["pairwise"])
    assert data["extra"] == "value"

def test_cache_get_by_key_miss(tmp_path):
    cache = PredictionCache(cache_dir=str(tmp_path), max_entries=2)
    data = cache.get_by_key("nonexistent_key")
    assert data is None

def test_cache_get_by_key_error_handling(tmp_path, caplog):
    cache = PredictionCache(cache_dir=str(tmp_path), max_entries=2)
    key = "error_key"
    results = {"val": 42}
    cache.set_by_key(key, results)

    import hashlib
    safe_key = hashlib.sha256(key.encode("utf-8")).hexdigest()
    cache_file = cache.cache_dir / f"{safe_key}.json"

    # Corrupt the JSON file to trigger Exception during load
    with open(cache_file, "w") as f:
        f.write("{invalid_json:")

    with caplog.at_level(logging.WARNING):
        data = cache.get_by_key(key)

    assert data is None
    assert "Failed to load prediction cache" in caplog.text

def test_cache_set_by_key_error_handling(tmp_path, caplog):
    cache = PredictionCache(cache_dir=str(tmp_path), max_entries=2)
    key = "error_key"
    results = {"val": 42}

    with patch("hashlib.sha256") as mock_sha256:
        # Create a mock object that returns a problematic string
        mock_hexdigest = mock_sha256.return_value
        mock_hexdigest.hexdigest.return_value = "nonexistent_dir/file"

        with caplog.at_level(logging.WARNING):
            cache.set_by_key(key, results)
        assert "Failed to save prediction cache" in caplog.text
