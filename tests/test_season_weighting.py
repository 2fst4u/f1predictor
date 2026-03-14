
import pandas as pd
from datetime import datetime, timezone, timedelta
from f1pred.features import compute_form_indices

def test_current_season_weight_boost():
    # Setup data: 1 race in 2023, 1 race in 2024
    # We want to see if boosting 2024 results actually changes the form index

    base_date = datetime(2024, 6, 1, tzinfo=timezone.utc)

    hist_data = [
        # 2023 Race: Winner (1st place, 25 pts)
        {"driverId": "max_verstappen", "season": 2023, "round": 1, "session": "race",
         "date": base_date - timedelta(days=365), "position": 1, "points": 25.0, "status": "Finished"},
        # 2024 Race: 10th place (1 pt)
        {"driverId": "max_verstappen", "season": 2024, "round": 1, "session": "race",
         "date": base_date - timedelta(days=10), "position": 10, "points": 1.0, "status": "Finished"},
    ]
    df = pd.DataFrame(hist_data)

    # Half life of 1000 days so exponential decay is minimal between these two races
    # (2024 result is still fresher, but not by much compared to 1000 days)
    half_life = 1000

    # 1. No boost (boost=1.0)
    form_no_boost = compute_form_indices(df, base_date, half_life, current_season=2024, boost_factor=1.0)
    score_no_boost = form_no_boost.loc[form_no_boost["driverId"] == "max_verstappen", "form_index"].iloc[0]

    # 2. Significant boost for 2024 (boost=10.0)
    form_boost = compute_form_indices(df, base_date, half_life, current_season=2024, boost_factor=10.0)
    score_boost = form_boost.loc[form_boost["driverId"] == "max_verstappen", "form_index"].iloc[0]

    # In 2023, score = -1 + 25 = 24
    # In 2024, score = -10 + 1 = -9
    # Without boost, the score should be between 24 and -9 (weighted average)
    # With boost=10.0 for 2024, the score should be much closer to -9

    print(f"Score no boost: {score_no_boost}")
    print(f"Score boost: {score_boost}")

    assert score_boost < score_no_boost

    # Verify it doesn't boost if season doesn't match
    form_wrong_season = compute_form_indices(df, base_date, half_life, current_season=2025, boost_factor=10.0)
    score_wrong_season = form_wrong_season.loc[form_wrong_season["driverId"] == "max_verstappen", "form_index"].iloc[0]

    assert score_wrong_season == score_no_boost


def test_build_hist_training_X_boost():
    from f1pred.models import build_hist_training_X

    base_date = datetime(2024, 6, 1, tzinfo=timezone.utc)

    # 2 events: one in 2023, one in 2024
    # When building training data for the 2024 event, it looks at prior history.
    # We can check if boosting 2024 results (if any) works, but usually prior history for a 2024 event
    # will be mostly previous years.
    # Let's have two 2024 events. E1 and E2. For E2, E1 is in the prior history.

    hist_data = []
    # Add 40 dummy races in 2022
    for i in range(40):
        hist_data.append({
            "driverId": "other", "season": 2022, "round": i+1, "session": "race",
            "date": base_date - timedelta(days=700 + i), "position": 10, "points": 1.0, "status": "Finished", "grid": 10
        })

    hist_data.extend([
        {"driverId": "max", "season": 2023, "round": 1, "session": "race",
         "date": base_date - timedelta(days=365), "position": 1, "points": 25.0, "status": "Finished", "grid": 1},
        {"driverId": "max", "season": 2024, "round": 1, "session": "race",
         "date": base_date - timedelta(days=10), "position": 10, "points": 1.0, "status": "Finished", "grid": 1},
        {"driverId": "max", "season": 2024, "round": 2, "session": "race",
         "date": base_date, "position": 5, "points": 10.0, "status": "Finished", "grid": 5},
    ])
    hist = pd.DataFrame(hist_data)
    X_current = pd.DataFrame([{"driverId": "max", "form_index": 0.0}])

    # For R2 (2024), prior is R1 (2023) and R1 (2024).
    # If boost=10.0, R1 (2024) should be boosted.

    # We can't easily see the internal form_index calculation result in build_hist_training_X
    # without mocking or checking the output.
    # The output of build_hist_training_X is a DataFrame where form_index is a column.

    res_no_boost = build_hist_training_X(hist, X_current, base_date + timedelta(days=1), boost_factor=1.0)
    # Get form_index for round 2
    # build_hist_training_X returns all events. We want the one for Round 2.
    # Wait, build_hist_training_X doesn't return the round. It just returns rows.
    # Let's just compare the whole thing.

    res_boost = build_hist_training_X(hist, X_current, base_date + timedelta(days=1), boost_factor=10.0)

    # The form_index in the output is the 'prior' form index.
    # For Round 2 (2024), the prior form index is calculated from Round 1 (2023) and Round 1 (2024).
    # Since Round 1 (2024) is 10th place and Round 1 (2023) is 1st place,
    # boosting Round 1 (2024) will LOWER the form_index for the Round 2 sample.

    res_no_boost = build_hist_training_X(hist, X_current, base_date + timedelta(days=1), boost_factor=1.0)
    res_boost = build_hist_training_X(hist, X_current, base_date + timedelta(days=1), boost_factor=10.0)

    fi_no_boost = res_no_boost[res_no_boost["grid"] == 5]["form_index"].iloc[0]
    fi_boost = res_boost[res_boost["grid"] == 5]["form_index"].iloc[0]

    assert fi_boost < fi_no_boost