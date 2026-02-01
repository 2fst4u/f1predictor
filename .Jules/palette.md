## 2026-01-31 - CLI Session Aliasing
**Learning:** Users often use shorthand for F1 sessions (e.g., "q" for Qualifying, "gp" for Race).
**Action:** Implement a normalization layer in `main.py` that maps aliases to canonical names and uses `difflib` for fuzzy suggestions on typos. This pattern can be applied to other CLI args like `--round`.

## 2026-02-05 - Color-Coded Accuracy in Tables
**Learning:** In prediction tables, showing just the actual value forces users to mentally calculate accuracy. Color-coding the "Actual" column based on delta (Green=Exact, Red=Miss) with subtle indicators (✓) provides instant feedback.
**Action:** When displaying predicted vs actual data, always use semantic coloring to highlight accuracy/error rather than just raw values.
