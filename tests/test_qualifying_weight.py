
import pandas as pd
from datetime import datetime, timezone, timedelta
from f1pred.models import build_hist_training_X

def test_build_hist_training_X_qual_boost():
    base_date = datetime(2024, 6, 1, tzinfo=timezone.utc)

    # Need 40 races to trigger training
    hist_data = []
    for i in range(40):
        hist_data.append({
            "driverId": "other", "season": 2022, "round": i+1, "session": "race",
            "date": base_date - timedelta(days=700 + i), "position": 10, "points": 1.0, "status": "Finished", "grid": 10, "qpos": 10
        })

    hist_data.extend([
        # Prior race for max so he is included as a sample
        {"driverId": "max", "season": 2022, "round": 1, "session": "race",
         "date": base_date - timedelta(days=800), "position": 1, "points": 25.0, "status": "Finished", "grid": 1},
        # 2023 Qual: 1st
        {"driverId": "max", "season": 2023, "round": 1, "session": "qualifying",
         "date": base_date - timedelta(days=365), "qpos": 1},
        # 2024 Qual: 10th
        {"driverId": "max", "season": 2024, "round": 1, "session": "qualifying",
         "date": base_date - timedelta(days=10), "qpos": 10},
        # 2024 Race (to provide a training sample)
        {"driverId": "max", "season": 2024, "round": 2, "session": "race",
         "date": base_date, "position": 5, "points": 10.0, "status": "Finished", "grid": 5},
    ])

    hist = pd.DataFrame(hist_data)
    X_current = pd.DataFrame([{"driverId": "max", "qualifying_form_index": 0.0}])

    # No qual boost
    res1 = build_hist_training_X(hist, X_current, base_date + timedelta(days=1), boost_factor=1.0, qual_boost_factor=1.0)
    assert res1 is not None
    qf1 = res1[res1["grid"] == 5]["qualifying_form_index"].iloc[0]

    # High qual boost for 2024
    res2 = build_hist_training_X(hist, X_current, base_date + timedelta(days=1), boost_factor=1.0, qual_boost_factor=10.0)
    assert res2 is not None
    qf2 = res2[res2["grid"] == 5]["qualifying_form_index"].iloc[0]

    print(f"Qual Form Index 1: {qf1}, Qual Form Index 2: {qf2}")
    assert qf2 < qf1 # More weight on 10th place should lower the score

def test_build_hist_training_X_sprint_boost():
    base_date = datetime(2024, 6, 1, tzinfo=timezone.utc)

    hist_data = []
    for i in range(40):
        hist_data.append({
            "driverId": "other", "season": 2022, "round": i+1, "session": "race",
            "date": base_date - timedelta(days=700 + i), "position": 10, "points": 1.0, "status": "Finished", "grid": 10, "qpos": 10
        })

    hist_data.extend([
        # 2023 Race: 1st (not boosted)
        {"driverId": "max", "season": 2023, "round": 1, "session": "race",
         "date": base_date - timedelta(days=365), "position": 1, "points": 25.0, "status": "Finished", "grid": 1},
        # 2024 Sprint: 10th (boosted)
        {"driverId": "max", "season": 2024, "round": 1, "session": "sprint",
         "date": base_date - timedelta(days=10), "position": 10, "points": 1.0, "status": "Finished", "grid": 1},
        # 2024 Race (to provide a training sample)
        {"driverId": "max", "season": 2024, "round": 2, "session": "race",
         "date": base_date, "position": 5, "points": 10.0, "status": "Finished", "grid": 5},
    ])

    hist = pd.DataFrame(hist_data)
    X_current = pd.DataFrame([{"driverId": "max", "form_index": 0.0}])

    # With high sprint_boost_factor, the sprint result should be boosted (lowering form_index)
    res1 = build_hist_training_X(hist, X_current, base_date + timedelta(days=1), boost_factor=1.0, qual_boost_factor=1.0, sprint_boost_factor=1.0)
    fi1 = res1[res1["grid"] == 5]["form_index"].iloc[0]

    res2 = build_hist_training_X(hist, X_current, base_date + timedelta(days=1), boost_factor=1.0, qual_boost_factor=100.0, sprint_boost_factor=100.0)
    fi2 = res2[res2["grid"] == 5]["form_index"].iloc[0]

    print(f"FI1: {fi1}, FI2: {fi2}")
    assert fi2 < fi1

if __name__ == "__main__":
    test_build_hist_training_X_qual_boost()
    test_build_hist_training_X_sprint_boost()
    print("Tests passed!")
