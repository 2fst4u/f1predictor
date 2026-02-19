## 2026-01-31 - CLI Session Aliasing
**Learning:** Users often use shorthand for F1 sessions (e.g., "q" for Qualifying, "gp" for Race).
**Action:** Implement a normalization layer in `main.py` that maps aliases to canonical names and uses `difflib` for fuzzy suggestions on typos. This pattern can be applied to other CLI args like `--round`.

## 2026-02-05 - Color-Coded Accuracy in Tables
**Learning:** In prediction tables, showing just the actual value forces users to mentally calculate accuracy. Color-coding the "Actual" column based on delta (Green=Exact, Red=Miss) with subtle indicators (✓) provides instant feedback.
**Action:** When displaying predicted vs actual data, always use semantic coloring to highlight accuracy/error rather than just raw values.

## 2026-02-09 - Environmental Context Scannability
**Learning:** Adding visual icons (☀️, ☁️, 💧) for environmental conditions in the CLI allows users to scan race context much faster than text alone.
**Action:** Use specific, recognizable emoji for key environmental variables alongside numerical data to improve "at a glance" readability.

## 2026-02-15 - Visual Hierarchy in Rankings
**Learning:** For ranked lists, color-coding the top positions (Gold/Silver/Bronze) provides immediate visual structure and celebrates success without needing cluttered icons or emojis.
**Action:** Use distinct colors for top 3 items in any ranked list to guide the eye.
