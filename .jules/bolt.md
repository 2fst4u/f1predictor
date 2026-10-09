## 2024-07-24 - Pandas Row Iteration Bottleneck
**Learning:** In codeblocks requiring row-by-row iteration over a DataFrame where dictionary-like semantics are needed (e.g., using `r.get()`), using `df.iterrows()` yields heavy Pandas Series objects that parse index labels repeatedly, causing massive overhead. Furthermore, doing O(N) boolean series lookups inside that loop (like `roster[roster["code"] == abbr]`) scales quadratically.
**Action:** Convert the iterated DataFrame into a list of lightweight dictionaries before looping using `df.to_dict("records")`. For foreign key lookups, build an O(1) dictionary map (e.g., `roster_map = dict(zip(df1["key"], df1["val"]))`) outside the loop to completely eliminate Pandas internal logic from the hot path.

## $(date +%Y-%m-%d) - Vectorized dictionary construction from DataFrames
**Learning:** In pandas, constructing a dictionary lookup map from a DataFrame by iterating over rows (e.g., `{row.id: (row.val1, row.val2) for row in df.itertuples()}`) is surprisingly slow due to the overhead of generating namedtuples for every row.
**Action:** Replace `itertuples()` comprehensions with a pure vectorized approach leveraging NumPy `.values` arrays: `dict(zip(df['id_col'].values, zip(df['col1'].values, df['col2'].values)))`. This approach skips Pandas' row abstraction entirely and yields a ~10x speedup, making it especially effective inside hot loops or frequent function calls.
## 2024-05-24 - Pandas `itertuples` with `.iloc` lookup optimization
**Learning:** In a loop iterating over `df.itertuples()`, looking up the row's values using `df.iloc[idx].get("col")` is extremely slow because it instantiates a Pandas Series inside the loop, incurring massive type-checking overhead.
**Action:** Use `getattr(row, "col", default)` to access the namedtuple properties directly, yielding an approximate 50x speedup in pure-Python overhead.
## 2024-05-24 - Pandas zip timezone tracking
**Learning:** When vectorizing slow `itertuples()` loops by extracting timezone-aware datetime columns using `df["date"].values` and `zip()`, the resulting NumPy array yields naive `datetime64[ns]` objects, stripping the timezone. Reconstructing a pandas Timestamp (`pd.Timestamp(d)`) without explicitly setting `tz="UTC"` causes fatal `TypeError: Cannot compare tz-naive and tz-aware datetime-like objects` exceptions later in the code.
**Action:** When extracting datetimes with `.values` in loops, always explicitly re-localize them if timezone awareness is required (e.g., `d = pd.Timestamp(evt_d); if d.tz is None: d = d.tz_localize("UTC")`).

## 2024-05-24 - Vectorized loop pandas extraction bottleneck
**Learning:** Extracting an array for iteration via `df["col"].values` completely removes pandas overhead, but doing `X_evt["grid"].values[idx]` *inside* an inner loop initiates a fresh pandas Series lookup and NumPy array slice for every single row, causing a massive performance degradation that makes it slower than `itertuples()`.
**Action:** Always extract the `.values` arrays for *all* required columns *outside* the loop and unpack them in the `zip()` expression to maintain O(1) Python iteration speed.
