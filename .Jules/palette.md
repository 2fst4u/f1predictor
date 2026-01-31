## 2026-01-31 - CLI Session Aliasing
**Learning:** Users often use shorthand for F1 sessions (e.g., "q" for Qualifying, "gp" for Race).
**Action:** Implement a normalization layer in `main.py` that maps aliases to canonical names and uses `difflib` for fuzzy suggestions on typos. This pattern can be applied to other CLI args like `--round`.
