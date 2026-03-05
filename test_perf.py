import pandas as pd
import numpy as np
import time

N = 10000
dates = pd.Series(pd.date_range("2000-01-01", periods=N, freq="h", tz="UTC"))
ref_ts = pd.Timestamp("2024-01-01", tz="UTC")

t0 = time.time()
for _ in range(100):
    ages1 = (ref_ts - dates).dt.total_seconds() / 86400.0
t1 = time.time()
print("dt.total_seconds():", t1 - t0)

NS_PER_DAY = 86400000000000.0
t0 = time.time()
for _ in range(100):
    diff = ref_ts.to_datetime64() - dates.values
    ages2 = diff.astype('timedelta64[ns]').astype(float) / NS_PER_DAY
t1 = time.time()
print("numpy diff:", t1 - t0)

print(np.allclose(ages1, ages2))
