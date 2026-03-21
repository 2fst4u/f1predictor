
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from f1pred.models import build_hist_training_X, train_pace_model

def test_shap_visibility():
    """Ensure that granular features are propagated through hist_X and appear in SHAP."""
    # Mock X_current with many features
    X_current = pd.DataFrame({
        "driverId": ["verstappen", "perez"],
        "constructorId": ["red_bull", "red_bull"],
        "form_index": [-1.0, -2.0],
        "qualifying_form_index": [-1.0, -2.0],
        "grid": [1, 2],
        "is_race": [1, 1],
        "is_qualifying": [0, 0],
        "is_sprint": [0, 0],
        "team_form_index": [10.0, 10.0],
        "circuit_avg_pos": [1.5, 4.0],
        "teammate_delta": [0.5, -0.5],
        "grid_finish_delta": [1.0, -1.0],
        "weather_effect": [0.1, -0.1]
    })

    # Mock hist with enough data to trigger build_hist_training_X
    # Needs at least 40 race rows
    hist_rows = []
    for i in range(50):
        hist_rows.append({
            "season": 2023,
            "round": (i % 25) + 1,
            "date": datetime(2023, 1, (i % 25) + 1, tzinfo=timezone.utc),
            "circuitId": "bahrain",
            "driverId": "verstappen" if i < 25 else "perez",
            "constructorId": "red_bull",
            "position": 1 if i < 25 else 2,
            "points": 25.0 if i < 25 else 18.0,
            "grid": 1 if i < 25 else 2,
            "session": "race",
            "qpos": 1 if i < 25 else 2
        })
    hist = pd.DataFrame(hist_rows)

    hist_X = build_hist_training_X(hist, X_current, datetime(2024, 1, 1, tzinfo=timezone.utc))

    assert hist_X is not None
    # Check that interesting features are populated in hist_X
    for col in ["team_form_index", "grid_finish_delta", "teammate_delta", "circuit_avg_pos"]:
        assert col in hist_X.columns, f"{col} missing from hist_X"
        assert not hist_X[col].isna().all(), f"{col} is all NaN in hist_X"

    # Train model
    pipe, pace_hat, features, shap_values = train_pace_model(X_current, "race", cfg=None, hist_X=hist_X)

    # Ensure granular features were used in training
    for col in ["team_form_index", "grid_finish_delta", "teammate_delta", "circuit_avg_pos"]:
        assert col in features, f"{col} missing from features used in training"

    # Ensure SHAP values contain at least one granular feature for the first driver
    if shap_values:
        hidden_feats = {"form_index", "qualifying_form_index", "grid", "is_race", "is_qualifying", "is_sprint"}
        visible = [k for k in shap_values[0].keys() if k not in hidden_feats]

        # We expect at least one of our granular features to have non-zero SHAP influence
        assert any(f in visible for f in ["team_form_index", "grid_finish_delta", "teammate_delta", "circuit_avg_pos"]), \
            f"None of the granular features are visible in SHAP. Visible: {visible}"
