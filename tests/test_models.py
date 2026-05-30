"""Tests for model training and prediction."""
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from f1pred.models import train_pace_model, estimate_dnf_probabilities, build_hist_training_X


def test_train_pace_model_basic(sample_features):
    """Test basic pace model training."""
    model, pace_hat, features, shap_vals = train_pace_model(sample_features, session_type='race')

    assert model is not None
    assert len(pace_hat) == len(sample_features)
    assert np.all(np.isfinite(pace_hat))
    assert isinstance(features, list)
    # shap_vals is None when shap is not installed; otherwise a list of dicts
    assert shap_vals is None or isinstance(shap_vals, list)


def test_train_pace_model_variance(sample_features):
    """Test that pace model produces varying predictions."""
    _, pace_hat, _, _ = train_pace_model(sample_features, session_type='race')

    # Pace predictions should have some variance (not all identical)
    assert np.std(pace_hat) > 0.01, "Pace predictions are too uniform"


def test_pace_model_order_makes_sense(sample_features):
    """Test that better form results in better (lower) pace."""
    # Set clear form differences: form_index 0=worst, 19=best
    # (form_index is HIGHER = BETTER: -position + points)
    sample_features['form_index'] = np.arange(len(sample_features), dtype=float)
    sample_features['team_form_index'] = 0.0

    _, pace_hat, _, _ = train_pace_model(sample_features, session_type='race')

    # Pace is LOWER = FASTER. Since we predict y = -form_index:
    # - Driver with form_index=19 (best) should have pace ~ -19 (lowest/fastest)
    # - Driver with form_index=0 (worst) should have pace ~ 0 (highest/slowest)
    # So drivers with lowest pace should have HIGHEST form_index
    order = np.argsort(pace_hat)  # Sorted from lowest (fastest) to highest (slowest)
    # First few in order (fastest) should have HIGHER form_index
    assert np.mean(sample_features.iloc[order[:5]]['form_index']) > \
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


def test_dnf_weights_are_normalized(sample_historical_data, sample_roster):
    """DNF probabilities must depend only on the *ratio* of driver/team weights,
    not their absolute magnitude (they are normalised to a convex blend)."""
    # Doubling both weights must not change the result (1,1 == 0.5,0.5 ratio).
    p_unit = estimate_dnf_probabilities(
        sample_historical_data, sample_roster, driver_weight=1.0, team_weight=1.0
    )
    p_half = estimate_dnf_probabilities(
        sample_historical_data, sample_roster, driver_weight=0.5, team_weight=0.5
    )
    assert np.allclose(p_unit, p_half)

    # Scaling a given ratio up must also be invariant (0.6/0.4 == 0.3/0.2).
    p_big = estimate_dnf_probabilities(
        sample_historical_data, sample_roster, driver_weight=0.6, team_weight=0.4
    )
    p_small = estimate_dnf_probabilities(
        sample_historical_data, sample_roster, driver_weight=0.3, team_weight=0.2
    )
    assert np.allclose(p_big, p_small)


def test_build_hist_training_X_basic():
    """Test that build_hist_training_X produces a valid training set from history."""
    n_events = 5
    rows = []
    base_date = datetime(2023, 3, 1, tzinfo=timezone.utc)
    for rnd in range(1, n_events + 1):
        event_date = base_date + timedelta(days=14 * rnd)
        for pos in range(1, 11):
            rows.append({
                "season": 2023, "round": rnd, "session": "race",
                "date": event_date,
                "driverId": f"driver_{pos}",
                "constructorId": f"team_{(pos - 1) // 2}",
                "position": pos,
                "points": max(0, 26 - pos * 2),
                "grid": pos,
                "status": "Finished",
            })
    hist = pd.DataFrame(rows)

    X_current = pd.DataFrame({
        "driverId": [f"driver_{i}" for i in range(1, 11)],
        "constructorId": [f"team_{(i - 1) // 2}" for i in range(1, 11)],
        "form_index": np.random.randn(10),
        "grid": list(range(1, 11)),
    })

    result = build_hist_training_X(hist, X_current, base_date + timedelta(days=100))
    assert result is not None
    assert "form_index" in result.columns
    assert len(result) > 0
    # All numeric columns from X_current should be present
    assert "grid" in result.columns


def test_build_hist_training_X_weather_effect(tmp_path):
    """With a populated weather cache and driver betas, the training matrix must
    gain a real, varying weather_effect column (instead of an all-NaN one)."""
    import json, os
    n_events = 6
    base_date = datetime(2023, 3, 1, tzinfo=timezone.utc)
    cache_dir = str(tmp_path)
    wdir = os.path.join(cache_dir, "weather")
    os.makedirs(wdir, exist_ok=True)

    rows = []
    for rnd in range(1, n_events + 1):
        event_date = base_date + timedelta(days=14 * rnd)
        # Per-event weather with rain that varies by round.
        with open(os.path.join(wdir, f"event_2023_{rnd}.json"), "w") as f:
            json.dump({"temp_mean": 20.0, "pressure_mean": 1010.0,
                       "wind_mean": 5.0, "rain_sum": float(rnd)}, f)
        for pos in range(1, 11):
            rows.append({
                "season": 2023, "round": rnd, "session": "race",
                "date": event_date, "driverId": f"driver_{pos}",
                "constructorId": f"team_{(pos - 1) // 2}", "position": pos,
                "points": max(0, 26 - pos * 2), "grid": pos, "status": "Finished",
            })
    hist = pd.DataFrame(rows)

    X_current = pd.DataFrame({
        "driverId": [f"driver_{i}" for i in range(1, 11)],
        "constructorId": [f"team_{(i - 1) // 2}" for i in range(1, 11)],
        "form_index": np.zeros(10),
        "weather_beta_temp": 0.0,
        "weather_beta_pressure": 0.0,
        "weather_beta_wind": 0.0,
        # Rain sensitivity varies by driver so weather_effect is non-degenerate.
        "weather_beta_rain": [0.1 * i for i in range(1, 11)],
    })
    ref_date = base_date + timedelta(days=14 * (n_events + 2))

    out = build_hist_training_X(hist, X_current, ref_date, cache_dir=cache_dir)
    assert out is not None
    assert "weather_effect" in out.columns
    # The feature must actually carry signal (beta_rain × rain_sum), not be flat.
    assert out["weather_effect"].notna().any()
    assert float(out["weather_effect"].std()) > 0.0

    # Without cache_dir the legacy behaviour is preserved (no weather_effect).
    out_legacy = build_hist_training_X(hist, X_current, ref_date)
    assert "weather_effect" not in out_legacy.columns


def test_build_hist_training_X_insufficient_history():
    """Test that build_hist_training_X returns None with insufficient data."""
    hist = pd.DataFrame({
        "season": [2023], "round": [1], "session": ["race"],
        "date": [datetime(2023, 3, 5, tzinfo=timezone.utc)],
        "driverId": ["driver_1"], "constructorId": ["team_1"],
        "position": [1], "points": [25], "grid": [1], "status": ["Finished"],
    })
    X_current = pd.DataFrame({"driverId": ["driver_1"], "form_index": [0.0]})
    result = build_hist_training_X(hist, X_current, datetime(2023, 6, 1, tzinfo=timezone.utc))
    assert result is None


def test_build_hist_training_X_none_history():
    """Test that build_hist_training_X handles None/empty history."""
    X_current = pd.DataFrame({"driverId": ["driver_1"], "form_index": [0.0]})
    assert build_hist_training_X(None, X_current, datetime.now(timezone.utc)) is None
    assert build_hist_training_X(pd.DataFrame(), X_current, datetime.now(timezone.utc)) is None


def test_train_pace_model_with_hist_X(sample_features):
    """Test that train_pace_model uses hist_X for out-of-sample training."""
    # Build a simple hist_X with same columns as sample_features
    hist_X = sample_features.copy()
    hist_X["form_index"] = np.linspace(-5, 5, len(hist_X))

    model, pace_hat, features, shap_vals = train_pace_model(
        sample_features, session_type="race", hist_X=hist_X,
    )
    assert model is not None
    assert len(pace_hat) == len(sample_features)
    assert np.all(np.isfinite(pace_hat))


def test_models_qual_team_avg_nan_handling():
    """Test that qualifying team average logic handles NaNs in round and constructorId."""
    from f1pred.models import build_hist_training_X
    import pandas as pd
    import numpy as np
    from datetime import datetime, timezone

    ref_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    hist = pd.DataFrame({
        "session": ["qualifying", "qualifying", "qualifying", "race", "race", "race"],
        "season": [2023, 2023, 2023, 2023, 2023, 2023],
        "round": [1, np.nan, 2, 1, 1, 2],
        "constructorId": ["c1", "c1", np.nan, "c1", "c1", "c2"],
        "driverId": ["d1", "d2", "d3", "d1", "d2", "d3"],
        "qpos": [2, 4, 6, 2, 4, 6],
        "position": [1, 2, 3, 1, 2, 3],
        "grid": [2, 4, 6, 2, 4, 6],
        "date": [ref_date, ref_date, ref_date, ref_date, ref_date, ref_date],
        "circuitId": ["circ1", "circ1", "circ2", "circ1", "circ1", "circ2"],
        "weather_rain": [0,0,0,0,0,0],
        "points": [1, 2, 3, 1, 2, 3]
    })

    X_current = pd.DataFrame({"driverId": ["d1", "d2", "d3"], "form_index": [0.0, 1.0, 2.0]})

    # NaN round/constructorId must not cause a negative-index crash in
    # np.bincount when computing qualifying team averages.
    res = build_hist_training_X(hist, X_current, ref_date)

    # With little history the builder may return None, but it must never raise
    # and, when it does return a matrix, it must be well-formed.
    assert res is None or isinstance(res, pd.DataFrame)
    if isinstance(res, pd.DataFrame):
        assert "driverId" in res.columns
        assert len(res) > 0
