import pandas as pd
import numpy as np
import time

# Create some dummy data
np.random.seed(0)
N = 100_000
df = pd.DataFrame({
    'date': pd.date_range("2000-01-01", periods=N, freq="15min", tz="UTC"),
    'position': np.random.randint(1, 21, size=N),
    'driverId': np.random.choice([f'd{i}' for i in range(25)], size=N),
})

ref_date = df['date'].max()

def run_pandas():
    dates = pd.to_datetime(df["date"], utc=True)
    ref_ts = pd.Timestamp(ref_date).tz_convert('UTC')
    ages = (ref_ts - dates).dt.total_seconds() / 86400.0
    w = np.exp2(-ages / max(1.0, 365.0))
    df["w"] = w
    df["weighted_pos"] = df["position"] * w
    grp = df.groupby("driverId").agg(
        w_pos_sum=("weighted_pos", "sum"),
        w_sum=("w", "sum")
    )
    res = {}
    for d, row in grp.iterrows():
        w_mean = row["w_pos_sum"] / max(1e-6, row["w_sum"])
        res[str(d)] = (20.0 - float(w_mean)) / 20.0
    return res

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


t0 = time.time()
r1 = run_pandas()
t1 = time.time()
print(f"Pandas: {t1-t0:.4f}s")

t0 = time.time()
r2 = run_numpy()
t1 = time.time()
print(f"Numpy/Vectorized: {t1-t0:.4f}s")
