## 2025-02-23 - Pandas groupby.sum() bottleneck in feature engineering
**Learning:** In pandas, repeatedly executing `df.groupby('col').sum()` on small to medium dataframes introduces significant Python overhead. Replacing this with `pd.factorize()` to generate integer bin codes followed by `np.bincount(codes, weights=values)` yields a 5x+ speedup. However, `pd.factorize()` explicitly encodes `NaN` values as `-1`, which causes `np.bincount` to crash with a `ValueError`.
**Action:** When migrating from `groupby` to `np.bincount`, always safeguard the factorization step by first doing an explicit `.dropna(subset=["col"])` on the grouping column to handle missing values exactly as `groupby` silently does.

## 2025-02-23 - Handle NaN values properly when replacing Pandas groupby aggregations with numpy bincount
**Learning:** When using `np.bincount` as a high-performance alternative to `groupby().agg(['sum', 'count'])`, extreme care must be taken regarding `NaN` values. Pandas `groupby.sum()` automatically skips `NaN` values, but `np.bincount(codes, weights=values)` will propagate a single `NaN` in the weights array to the entire sum for that bin, destroying the metric.
**Action:** When calculating weighted sums or counts via bincount, you MUST perform an explicit `.dropna(subset=[group_col, weight_col])` before factorizing and applying bincount.
