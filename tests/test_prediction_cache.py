
import os
import shutil
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from f1pred.util import PredictionCache
from f1pred.config import load_config
from f1pred.predict import run_predictions_for_event

@pytest.fixture
def cache_dir(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    return str(d)

def test_prediction_cache_basic(cache_dir):
    pc = PredictionCache(cache_dir, max_entries=5)

    inputs = {"season": 2024, "round": 1, "session": "race", "weather": {"temp": 25.0}}
    results = {
        "ranked": pd.DataFrame({"driverId": ["verstappen", "perez"], "predicted_position": [1, 2]}),
        "prob_matrix": np.array([[0.8, 0.1], [0.1, 0.8]]),
        "weather": {"temp": 25.0}
    }

    # Cache miss
    assert pc.get(inputs) is None

    # Set and get
    pc.set(inputs, results)
    cached = pc.get(inputs)
    assert cached is not None
    pd.testing.assert_frame_equal(cached["ranked"], results["ranked"])
    np.testing.assert_array_equal(cached["prob_matrix"], results["prob_matrix"])
    assert cached["weather"] == results["weather"]

def test_prediction_cache_rolling(cache_dir):
    pc = PredictionCache(cache_dir, max_entries=2)

    pc.set({"id": 1}, {"res": 1})
    pc.set({"id": 2}, {"res": 2})

    # Should still have both
    assert pc.get({"id": 1}) is not None
    assert pc.get({"id": 2}) is not None

    # Set third, should trigger cleanup of oldest (which is now id 1 because we just 'get' id 2)
    # Wait, pc.get(id 1) touched it last. So id 2 is oldest.
    # Order: set 1 (1), set 2 (1, 2), get 1 (2, 1), get 2 (1, 2).
    # Let's be explicit.

    pc.set({"id": 1}, {"res": 1}) # mtime(1)
    import time
    time.sleep(0.1)
    pc.set({"id": 2}, {"res": 2}) # mtime(2)
    time.sleep(0.1)

    pc.set({"id": 3}, {"res": 3}) # should delete 1

    assert pc.get({"id": 1}) is None
    assert pc.get({"id": 2}) is not None
    assert pc.get({"id": 3}) is not None

def test_prediction_cache_input_sensitivity(cache_dir):
    pc = PredictionCache(cache_dir)

    inputs1 = {"weather": {"temp": 25.0}}
    results1 = {"val": 1}

    inputs2 = {"weather": {"temp": 26.0}}
    results2 = {"val": 2}

    pc.set(inputs1, results1)
    pc.set(inputs2, results2)

    assert pc.get(inputs1)["val"] == 1
    assert pc.get(inputs2)["val"] == 2
