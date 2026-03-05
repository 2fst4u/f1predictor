import pandas as pd
import numpy as np
import time

def run_numpy():
    # simulate dates as datetime series
    dates = pd.Series(pd.date_range("2000-01-01", periods=10_000, freq="15min", tz="UTC"))
    ref_ts = pd.Timestamp("2024-01-01", tz="UTC")

    t0 = time.time()
    for _ in range(100):
        # 1. the dt.total_seconds() way
        ages = (ref_ts - dates).dt.total_seconds() / 86400.0
    t1 = time.time()
    print("dt.total_seconds() time:", t1 - t0)

    t0 = time.time()
    for _ in range(100):
        # 2. the numpy way
        diff = ref_ts.to_datetime64() - dates.values
        ages2 = diff.astype('timedelta64[ns]').astype(float) / 86400000000000.0
    t1 = time.time()
    print("numpy way time:", t1 - t0)

run_numpy()
