import pandas as pd
import numpy as np
import time

def run_numpy():
    dates = pd.to_datetime(df["date"], utc=True)
    ref_ts = pd.Timestamp(ref_date).tz_convert('UTC')
    # Use numpy datetime64
    diff = ref_ts.to_datetime64() - dates.values
    ages = diff.astype('timedelta64[ns]').astype(float) / 86400000000000.0
    w = np.exp2(-ages / max(1.0, 365.0))
    df["w"] = w
    df["weighted_pos"] = df["position"] * w
    grp = df.groupby("driverId").agg(
        w_pos_sum=("weighted_pos", "sum"),
        w_sum=("w", "sum")
    )
    # Instead of iterrows
    w_mean = grp["w_pos_sum"] / grp["w_sum"].clip(lower=1e-6)
    res = ((20.0 - w_mean) / 20.0).to_dict()
    return {str(k): float(v) for k, v in res.items()}

# Compare pandas datetime parsing vs raw values if dates is already datetime64
