import pandas as pd
import numpy as np
import time

np.random.seed(0)
N = 100_000
df = pd.DataFrame({
    'date': pd.date_range("2000-01-01", periods=N, freq="15min", tz="UTC"),
    'position': np.random.randint(1, 21, size=N),
    'driverId': np.random.choice([f'd{i}' for i in range(25)], size=N),
})

ref_date = df['date'].max()

def run_pandas_iterrows():
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

def run_vectorized():
    # Optimization: Direct numpy operations on datetime64[ns] are ~6x faster than .dt.total_seconds()
    # Also vectorize iterrows into pandas dict cast
    dates = pd.to_datetime(df["date"], utc=True)
    ref_ts = pd.Timestamp(ref_date).tz_convert('UTC')
    diff = ref_ts.to_datetime64() - dates.values
    ages = diff.astype('timedelta64[ns]').astype(float) / 86400000000000.0
    w = np.exp2(-ages / max(1.0, 365.0))
    df["w"] = w
    df["weighted_pos"] = df["position"] * w
    grp = df.groupby("driverId").agg(
        w_pos_sum=("weighted_pos", "sum"),
        w_sum=("w", "sum")
    )
    w_mean = grp["w_pos_sum"] / grp["w_sum"].clip(lower=1e-6)
    res = ((20.0 - w_mean) / 20.0).to_dict()
    return {str(k): float(v) for k, v in res.items()}

# Benchmark with smaller N to see iterrows penalty
N2 = 1000
df2 = pd.DataFrame({
    'date': pd.date_range("2000-01-01", periods=N2, freq="1d", tz="UTC"),
    'position': np.random.randint(1, 21, size=N2),
    'driverId': np.random.choice([f'd{i}' for i in range(25)], size=N2),
})

def time_it(func, df_in, iters=100):
    global df
    df = df_in
    t0 = time.time()
    for _ in range(iters):
        func()
    return time.time() - t0

print(f"Iterrows approach: {time_it(run_pandas_iterrows, df2):.4f}s")
print(f"Vectorized approach: {time_it(run_vectorized, df2):.4f}s")
