import pandas as pd
import numpy as np
import time

def run_numpy():
    # simulate dates as datetime series
    df = pd.DataFrame({
        'driverId': np.random.choice([f'd{i}' for i in range(25)], size=10_000),
        'weighted_pos': np.random.randn(10_000),
        'w': np.random.rand(10_000)
    })

    # method 1: the iterrows way
    t0 = time.time()
    for _ in range(100):
        grp = df.groupby("driverId").agg(
            w_pos_sum=("weighted_pos", "sum"),
            w_sum=("w", "sum")
        )
        res1 = {}
        for d, row in grp.iterrows():
            w_mean = row["w_pos_sum"] / max(1e-6, row["w_sum"])
            # smaller average position => higher strength
            res1[str(d)] = (20.0 - float(w_mean)) / 20.0
    t1 = time.time()
    print("iterrows time:", t1 - t0)

    # method 2: the vectorized way
    t0 = time.time()
    for _ in range(100):
        grp = df.groupby("driverId").agg(
            w_pos_sum=("weighted_pos", "sum"),
            w_sum=("w", "sum")
        )
        w_mean = grp["w_pos_sum"] / grp["w_sum"].clip(lower=1e-6)
        res2 = ((20.0 - w_mean) / 20.0).to_dict()
        res2 = {str(k): float(v) for k, v in res2.items()}
    t1 = time.time()
    print("vectorized time:", t1 - t0)

    for k in res1:
        assert np.isclose(res1[k], res2[k])

run_numpy()
