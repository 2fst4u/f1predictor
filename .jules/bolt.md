## 2024-05-22 - Vectorized Weighted Averages
**Learning:** `groupby().apply()` for weighted averages is significantly slower (10x-100x) than computing weighted sums and dividing vectors. However, replacing `max(1e-6, sum)` with simple division requires care.
**Action:** Use `.clip(lower=1e-6)` on the denominator series (e.g., `sums["w"].clip(lower=1e-6)`) to strictly preserve the original "clamp" behavior that handles zero or near-zero weights, rather than `.replace(0, 1e-6)` which changes the result for small non-zero values.

## 2026-01-13 - Vectorized Grouped Correlations
**Learning:** Computing correlations inside a Python loop over `groupby` groups (e.g., `for name, group in df.groupby(): ... np.corrcoef(...)`) is significantly slower than using the vectorized `df.groupby().corr()` method, even when filtering specific columns.
**Action:** Use `df.groupby().corr()` to compute the full correlation matrix and then slice the result using `.xs()` or similar indexing to retrieve the specific correlations needed. Be mindful that `corr()` returns `NaN` for constant columns (std dev = 0), whereas manual logic might return 0.0, so explicit `fillna(0.0)` may be required to match legacy behavior.

## 2026-01-14 - Vectorized Lookup via Merge
**Learning:** Iterating over a DataFrame (`iterrows`) to map values from a dictionary (e.g., weather data per event) is significantly slower than creating a DataFrame from the dictionary and using `.merge()`.
**Action:** Always prefer `pd.DataFrame(dict_list).merge(target_df)` over loop-based mapping when enriching a DataFrame with external data keyed by multiple columns.

## 2026-01-14 - Vectorized Pairwise Metrics
**Learning:** Computing pairwise metrics (like Brier score) using nested Python loops (`for i in range(n): for j in range(i+1, n):`) is O(n²) and slow for repeated calls.
**Action:** Use NumPy broadcasting to create a pairwise comparison matrix (e.g., `Y = val[:, None] < val[None, :]`) and compute differences/errors on the full matrix. Then use `np.triu_indices` to extract the unique pairs (upper triangle) for the final aggregation.

## 2026-01-14 - Conditional Expensive Computations
**Learning:** Even optimized vectorized O(N²) operations (like pairwise matrices in Monte Carlo simulations) incur measurable overhead (~30% of function time).
**Action:** Use a flag (e.g., `compute_pairwise=False`) to skip expensive auxiliary calculations when their results are not consumed by the caller.

## 2026-01-14 - Memory Bandwidth in Broadcasting
**Learning:** When broadcasting comparisons for large arrays (e.g. `(Draws, N, 1) < (Draws, 1, N)`), using smaller integer types (e.g. `int16` vs default `int64`) for the source data can significantly reduce memory bandwidth and improve performance (~30% speedup), even if the result is the same boolean mask.
**Action:** Cast integer arrays to the smallest sufficient type (e.g. `astype(np.int16)`) before performing heavy broadcasting operations.

## 2025-02-18 - Caching > Micro-Optimization
**Learning:** Sometimes the best optimization isn't vectorization but caching. `EloModel.fit` was O(N²) and took ~0.7s per session. Vectorizing the inner loop only saved ~0.05s due to Python overhead in the outer loop (groupby). Caching the fitted model across sessions (since history is identical for an event) saved ~3.5s per event.
**Action:** Always check if a heavy computation is repeated with identical inputs before trying to optimize the computation itself.
