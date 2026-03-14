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

## 2026-05-22 - Parallelizing IO-Bound Requests
**Learning:** Sequential HTTP requests in a loop (even with caching) are a major bottleneck for cold cache scenarios.
**Action:** Use `concurrent.futures.ThreadPoolExecutor` to parallelize IO-bound tasks like fetching historical weather data, yielding significant speedups (~9x in benchmarks).

## 2026-05-22 - Parallelizing Deduplication Regression
**Learning:** When moving from immediate sequential processing to batch processing (e.g., collecting tasks for a thread pool), checks against the results cache inside the loop become ineffective because the cache isn't updated until tasks complete.
**Action:** Use a local `seen` set to track items processed *within* the current batch generation loop to restore deduplication logic.

## 2026-05-23 - Granular Parallelization
**Learning:** Even when a top-level loop (e.g., iterating years) has dependencies that prevent full parallelization (e.g., early exit based on data), parallelizing independent IO operations *within* the loop body (e.g., race/qual/sprint fetch) still yields significant gains (3x speedup).
**Action:** Look for clusters of independent IO calls within sequential loops and wrap them in a local `ThreadPoolExecutor`.

## 2026-05-24 - Event-Level Caching
**Learning:** When iterating over sub-components of a larger entity (e.g., sessions in an event), identifying data that is constant across the entity and lifting its retrieval out of the loop prevents redundant I/O and processing.
**Action:** Identify constant data (like `roster` for an event) and fetch it once, passing it to sub-routines via an optional override argument to bypass redundant calculations.

## 2026-05-25 - Numpy Datetime Arithmetic
**Learning:** Computing date differences (ages) using pandas `.dt.days` accessor on a Series is significantly slower (~6x) than converting to numpy `datetime64[ns]` and performing direct arithmetic, due to pandas overhead.
**Action:** For heavy date difference calculations, convert inputs to `datetime64[ns]` (e.g., `pd.Timestamp(ref).to_datetime64() - series.values`) and divide by nanoseconds per day/unit.

## 2024-06-13 - [Monte Carlo Pairwise Optimization]
**Learning:** In NumPy, avoiding large N-dimensional boolean intermediate arrays (like a 3D `(draws, N, N)` array) in favor of looping over a smaller dimension (like $N=20$) and doing 2D vectorized slices can save significant memory bandwidth and CPU cache misses, leading to a ~40% speedup in Monte Carlo probability generation.
**Action:** When vectorizing across multiple dimensions, if one dimension is small (e.g. $N \le 20$), consider a Python loop over that dimension rather than full N-D broadcasting.

## 2026-05-26 - Optimizer Objective Precomputation
**Learning:** Pandas indexing and extracting groupings (e.g. `np.unique` to find grouped indices) inside a heavily called `scipy.optimize.minimize` objective function causes severe python overhead.
**Action:** Hoist data extraction (converting `pd.DataFrame` columns to `.values`) and grouping operations (generating indices/masks for grouping logic) outside the `objective` function into NumPy arrays before optimization starts, which reduces objective calculation times dramatically (e.g. ~2.8x speedup).

## 2026-05-27 - Vectorized Standardization Grouping
**Learning:** In optimization loops calling an objective function hundreds of times, a Python `for` loop slicing data arrays and calculating standardizations (mean/std) per-group becomes a massive bottleneck.
**Action:** Use pure NumPy `np.bincount` to vectorize sum and sum-of-squares aggregations over grouped integer indices. This calculates per-group means and variances simultaneously across the entire array, broadcasting results back via index arrays, yielding ~250x speedups over iterative masking.

## 2026-05-28 - Numpy standard_normal over normal
**Learning:** In NumPy Monte Carlo simulations, generating noise using `rng.standard_normal(size) * scale` is noticeably faster than `rng.normal(0, scale, size)` because it bypasses the internal Python-to-C parsing and validation overhead associated with the mean/scale parameters in the `normal` function.
**Action:** For high-throughput simulations, always prefer `standard_normal` multiplied by the scale scalar.

## 2026-05-28 - Transposed List Comprehensions vs Slice Assignments
**Learning:** When computing independent 1D aggregations across columns of a 2D matrix (like running `np.bincount` per column), it is ~10-20% faster to collect them in a list comprehension, wrap them in a 2D NumPy array, and transpose (`np.array([...]).T`), rather than iteratively assigning each column slice to a pre-allocated matrix (`counts[:, p] = ...`).
**Action:** Use transposed list comprehensions to construct 2D aggregates from 1D operations instead of iteratively updating a pre-allocated zeros array.

## 2026-05-28 - default_rng Breaking Reproducibility
**Learning:** While `np.random.default_rng()` is significantly faster than the legacy `np.random.RandomState()` (due to its PCG64 algorithm), swapping them in legacy codebases inherently breaks exact reproducible sequences, rendering any strict reproducible test suites (or user expectations of seeded stability) broken.
**Action:** Do not blindly upgrade `RandomState` to `default_rng` in performance PRs without explicitly warning that sequence stability will be broken. In strict environments, optimize around `RandomState` directly.

## 2026-05-29 - Vectorizing Model Fitting with np.bincount over GroupBy
**Learning:** When optimizing models like `BradleyTerryModel.fit` or `MixedEffectsLikeModel.fit` that rely on group aggregations (e.g., driver and team effects), `pandas` `groupby().agg()` introduces severe Python looping overhead inside the dataframe manipulation.
**Action:** Instead of `groupby`, use `np.unique(..., return_inverse=True)` to map categorical keys to continuous integer indices, then apply `np.bincount(idx, weights=val)` to perform sum and weighted-sum aggregations in highly optimized C-code. This reduces typical method time significantly without altering the mathematical behavior.
## 2024-05-18 - Replacing pandas groupby in loops with np.bincount
**Learning:** Performing `pandas.groupby().agg()` operations on historical data inside an iterative event loop (such as the out-of-sample data builder in `f1pred/models.py`) causes severe overhead. When this needs to run hundreds of times per CLI command, pandas' validation and grouping logic dominates runtime.
**Action:** When performing grouped aggregations inside a tight iterative loop, pre-extract pandas columns to pure NumPy arrays and use `pd.factorize()` once outside the loop to map group IDs (e.g. `driverId`) to integers. Then, inside the loop, use `np.bincount` to perform sums and counts, yielding 10x-15x speedups.
