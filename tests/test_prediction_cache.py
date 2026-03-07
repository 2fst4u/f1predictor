import pandas as pd
import numpy as np
from f1pred.util import PredictionCache

def test_prediction_cache_basic(tmp_path):
    cache_dir = tmp_path / "cache"
    cache = PredictionCache(str(cache_dir), max_entries=3)

    inputs = {"a": 1, "b": "test", "c": [1, 2, 3]}
    results = {"score": 0.95, "data": [10, 20]}

    # Miss
    assert cache.get(inputs) is None

    # Set
    cache.set(inputs, results)

    # Hit
    hit = cache.get(inputs)
    assert hit == results

def test_prediction_cache_complex_types(tmp_path):
    cache_dir = tmp_path / "cache"
    cache = PredictionCache(str(cache_dir))

    df = pd.DataFrame({"driver": ["VER", "NOR"], "pos": [1, 2]})
    arr = np.array([0.1, 0.2, 0.7])

    inputs = {"X": df, "weights": {"w1": 0.5}}
    results = {
        "ranked": df,
        "prob_matrix": arr,
        "meta": {"weather": "sunny"}
    }

    cache.set(inputs, results)
    hit = cache.get(inputs)

    assert hit is not None
    pd.testing.assert_frame_equal(hit["ranked"], df)
    np.testing.assert_array_equal(hit["prob_matrix"], arr)
    assert hit["meta"] == results["meta"]

def test_prediction_cache_rolling(tmp_path):
    cache_dir = tmp_path / "cache"
    # Small cache to test rolling deletion
    cache = PredictionCache(str(cache_dir), max_entries=2)

    inputs1 = {"v": 1}
    inputs2 = {"v": 2}
    inputs3 = {"v": 3}
    res = {"ok": True}

    cache.set(inputs1, res)
    import time
    time.sleep(1.1) # Ensure different mtimes
    cache.set(inputs2, res)
    time.sleep(1.1)

    # Touch inputs2 then inputs1
    assert cache.get(inputs2) is not None
    time.sleep(1.1)
    assert cache.get(inputs1) is not None
    time.sleep(1.1)

    # This should trigger deletion of the oldest.
    # Since we touched inputs1 last via get(), inputs2 is now oldest.
    cache.set(inputs3, res)

    assert cache.get(inputs3) is not None
    assert cache.get(inputs1) is not None
    assert cache.get(inputs2) is None # Should have been evicted

def test_prediction_cache_key_stability(tmp_path):
    cache_dir = tmp_path / "cache"
    cache = PredictionCache(str(cache_dir))

    # Inputs with different order but same content
    inputs1 = {"a": 1, "b": 2}
    inputs2 = {"b": 2, "a": 1}

    res = {"val": 42}
    cache.set(inputs1, res)
    assert cache.get(inputs2) == res
