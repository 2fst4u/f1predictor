from f1pred.util import PredictionCache
import os
import numpy as np
import pandas as pd

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
