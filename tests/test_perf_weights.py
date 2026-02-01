
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta, timezone
from f1pred.features import exponential_weights

def test_exponential_weights_correctness():
    ref_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    # Create dates: 0 days ago, 1 day ago, 10 days ago
    dates_list = [
        ref_date,
        ref_date - timedelta(days=1),
        ref_date - timedelta(days=10)
    ]
    dates_series = pd.Series(dates_list)

    half_life = 10

    weights = exponential_weights(dates_series, ref_date, half_life)

    # Expected:
    # 0 days -> exp2(-0/10) = 1.0
    # 1 day -> exp2(-1/10) = 0.933...
    # 10 days -> exp2(-10/10) = 0.5

    assert np.isclose(weights[0], 1.0)
    assert np.isclose(weights[1], 2**(-0.1))
    assert np.isclose(weights[2], 0.5)

def test_exponential_weights_nat():
    ref_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    dates_list = [pd.NaT, ref_date]
    dates_series = pd.Series(dates_list)

    weights = exponential_weights(dates_series, ref_date, 10)

    # NaT should be treated as 0 age -> weight 1.0 (based on fillna(0) behavior)
    assert np.isclose(weights[0], 1.0)
    assert np.isclose(weights[1], 1.0)

def test_exponential_weights_list_input():
    ref_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    dates_list = [
        ref_date,
        ref_date - timedelta(days=10)
    ]

    weights = exponential_weights(dates_list, ref_date, 10)

    assert np.isclose(weights[0], 1.0)
    assert np.isclose(weights[1], 0.5)
