## 2024-05-22 - Vectorized Weighted Averages
**Learning:** `groupby().apply()` for weighted averages is significantly slower (10x-100x) than computing weighted sums and dividing vectors. However, replacing `max(1e-6, sum)` with simple division requires care.
**Action:** Use `.clip(lower=1e-6)` on the denominator series (e.g., `sums["w"].clip(lower=1e-6)`) to strictly preserve the original "clamp" behavior that handles zero or near-zero weights, rather than `.replace(0, 1e-6)` which changes the result for small non-zero values.
