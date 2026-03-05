import pandas as pd
import numpy as np
import time

N = 10000
dates = pd.Series(pd.date_range("2000-01-01", periods=N, freq="h", tz="UTC"))
dates_naive = pd.Series(pd.date_range("2000-01-01", periods=N, freq="h"))
ref_ts = pd.Timestamp("2024-01-01", tz="UTC")
ref_ts_naive = pd.Timestamp("2024-01-01")

# Try with tz-aware
try:
    diff = ref_ts.to_datetime64() - dates.values
    ages2 = diff.astype('timedelta64[ns]').astype(float) / 86400000000000.0
    print("tz-aware worked directly:", len(ages2))
except Exception as e:
    print("tz-aware failed:", e)

# Convert to naive first
try:
    diff = ref_ts.tz_convert(None).to_datetime64() - dates.dt.tz_localize(None).values
    ages2 = diff.astype('timedelta64[ns]').astype(float) / 86400000000000.0
    print("tz-localize None worked:", len(ages2))
except Exception as e:
    print("tz-localize None failed:", e)
