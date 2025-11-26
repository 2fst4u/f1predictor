"""Tests for model training and prediction."""
import pytest
import numpy as np
import pandas as pd
from f1pred.models import train_pace_model, estimate_dnf_probabilities


def test_train_pace_model_basic(sample_features):
    """Test basic pace model training."""
    model, pace_hat, features = train_pace_model(sample_features, session_type='race')

    assert model is not None
    assert len(pace_hat) == len(sample_features)
    assert np.all(np.isfinite(pace_hat))
    assert isinstance(features, list)


def test_train_pace_model_variance(sample_features):
    """Test that pace model produces varying predictions."""
    _, pace_hat, _ = train_pace_model(sample_features, session_type='race')

    # Pace predictions should have some variance (not all identical)
    assert np.std(pace_hat) > 0.01, "Pace predictions are too uniform"


def test_pace_model_order_makes_sense(sample_features):
    """Test that better form results in better pace."""
    # Set clear form differences
    sample_features['form_index'] = np.arange(len(sample_features), dtype=float)
    sample_features['team_form_index'] = 0.0
    sample_features['driver_team_form_index'] = 0.0

    _, pace_hat, _ = train_pace_model(sample_features, session_type='race')

    # Lower pace index should correlate with better (lower) form_index
    # Since we're predicting -form_index, lower predicted values should
    # correspond to better drivers
    order = np.argsort(pace_hat)
    # First few in order should have lower form_index
    assert np.mean(sample_features.iloc[order[:5]]['form_index']) < \
           np.mean(sample_features.iloc[order[-5:]]['form_index'])


def test_estimate_dnf_probabilities(sample_historical_data, sample_roster):
    """Test DNF probability estimation."""
    dnf_probs = estimate_dnf_probabilities(sample_historical_data, sample_roster)

    assert len(dnf_probs) == len(sample_roster)
    assert np.all(dnf_probs >= 0.0)
    assert np.all(dnf_probs <= 1.0)
    assert np.all(np.isfinite(dnf_probs))


def test_dnf_probabilities_reasonable_range(sample_historical_data, sample_roster):
    """Test that DNF probabilities are in a reasonable range."""
    dnf_probs = estimate_dnf_probabilities(sample_historical_data, sample_roster)

    # Should be clipped to reasonable range
    assert np.min(dnf_probs) >= 0.02
    assert np.max(dnf_probs) <= 0.35
