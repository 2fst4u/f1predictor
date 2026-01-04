"""Tests for feature engineering."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from f1pred.features import (
    exponential_weights,
    compute_form_indices,
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


# Note: teammate_head_to_head was removed in a refactor; test removed
