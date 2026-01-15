## 2024-05-22 - Vectorized Weighted Averages
**Learning:** `groupby().apply()` for weighted averages is significantly slower (10x-100x) than computing weighted sums and dividing vectors. However, replacing `max(1e-6, sum)` with simple division requires care.
**Action:** Use `.clip(lower=1e-6)` on the denominator series (e.g., `sums["w"].clip(lower=1e-6)`) to strictly preserve the original "clamp" behavior that handles zero or near-zero weights, rather than `.replace(0, 1e-6)` which changes the result for small non-zero values.

## 2026-01-13 - Vectorized Grouped Correlations
**Learning:** Computing correlations inside a Python loop over `groupby` groups (e.g., `for name, group in df.groupby(): ... np.corrcoef(...)`) is significantly slower than using the vectorized `df.groupby().corr()` method, even when filtering specific columns.
**Action:** Use `df.groupby().corr()` to compute the full correlation matrix and then slice the result using `.xs()` or similar indexing to retrieve the specific correlations needed. Be mindful that `corr()` returns `NaN` for constant columns (std dev = 0), whereas manual logic might return 0.0, so explicit `fillna(0.0)` may be required to match legacy behavior.

## 2026-01-14 - Vectorized Lookup via Merge
**Learning:** Iterating over a DataFrame (`iterrows`) to map values from a dictionary (e.g., weather data per event) is significantly slower than creating a DataFrame from the dictionary and using `.merge()`.
**Action:** Always prefer `pd.DataFrame(dict_list).merge(target_df)` over loop-based mapping when enriching a DataFrame with external data keyed by multiple columns.
