"""Tests for feature engineering."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from f1pred.features import (
    exponential_weights,
    compute_form_indices,
    teammate_head_to_head,
)


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


def test_teammate_head_to_head():
    """Test teammate head-to-head calculation."""
    # Create qualifying data with teammates
    data = pd.DataFrame({
        'season': [2023] * 4,
        'round': [1] * 4,
        'session': ['qualifying'] * 4,
        'date': [datetime(2023, 3, 5, tzinfo=timezone.utc)] * 4,
        'driverId': ['hamilton', 'russell', 'verstappen', 'perez'],
        'constructorId': ['mercedes', 'mercedes', 'red_bull', 'red_bull'],
        'qpos': [3.0, 5.0, 1.0, 2.0],
    })

    h2h = teammate_head_to_head(data)

    assert 'driverA' in h2h.columns
    assert 'driverB' in h2h.columns
    assert 'a_better' in h2h.columns
    # Should have 2 teammate pairs
    assert len(h2h) == 2
