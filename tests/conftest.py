"""Pytest configuration and fixtures."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone


@pytest.fixture
def sample_historical_data():
    """Generate sample historical race data."""
    return pd.DataFrame({
        'season': [2023] * 20,
        'round': [1] * 20,
        'session': ['race'] * 20,
        'date': [datetime(2023, 3, 5, tzinfo=timezone.utc)] * 20,
        'driverId': [f'driver_{i}' for i in range(20)],
        'constructorId': [f'team_{i//2}' for i in range(20)],
        'position': list(range(1, 21)),
        'points': [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 10,
        'status': ['Finished'] * 18 + ['DNF', 'DNF'],
    })


@pytest.fixture
def sample_roster():
    """Generate sample driver roster."""
    return pd.DataFrame({
        'driverId': [f'driver_{i}' for i in range(20)],
        'name': [f'Driver {i}' for i in range(20)],
        'code': [f'DR{i}' for i in range(20)],
        'constructorId': [f'team_{i//2}' for i in range(20)],
        'constructorName': [f'Team {i//2}' for i in range(20)],
    })


@pytest.fixture
def sample_features():
    """Generate sample feature matrix."""
    n_drivers = 20
    return pd.DataFrame({
        'driverId': [f'driver_{i}' for i in range(n_drivers)],
        'constructorId': [f'team_{i//2}' for i in range(n_drivers)],
        'form_index': np.random.randn(n_drivers),
        'team_form_index': np.random.randn(n_drivers),
        'driver_team_form_index': np.random.randn(n_drivers),
        'weather_beta_temp': np.random.randn(n_drivers) * 0.1,
        'weather_beta_rain': np.random.randn(n_drivers) * 0.1,
        'session_type': ['race'] * n_drivers,
    })
