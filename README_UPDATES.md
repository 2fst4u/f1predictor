# README Updates - Add These Sections

Add these sections to your README.md:

## Badges

Add at the top of README.md after the title:

```markdown
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
```

## Quick Commands

Add after Installation section:

```markdown
## Quick Commands

For convenience, you can use the Makefile:

```bash
make install   # Install dependencies
make predict   # Run prediction for next race
make backtest  # Run backtesting
make clean     # Clear cache
```

Or run directly:

```bash
python main.py --round next --html
```
```

## Troubleshooting

Add before the Extending section:

```markdown
## Troubleshooting

### Predictions seem random or uniform

**Solution:** Clear the cache and re-run:
```bash
make clean
# or
rm -rf .cache/
python main.py --round next
```

The cache may contain outdated feature calculations from before the v1.1.0 fixes.

### Low variance warnings in logs

If you see warnings like "Pace predictions have very low variance":
- This can happen for events with limited historical data
- The system will add small noise to break ties
- Results will improve as more historical data is available

### Import errors for LightGBM

**On macOS (Intel):**
```bash
pip uninstall lightgbm
pip install lightgbm --no-binary lightgbm
```

**Alternative:** The system will fall back to XGBoost or scikit-learn if LightGBM is unavailable.

For more details, see [FIXES_AND_IMPROVEMENTS.md](FIXES_AND_IMPROVEMENTS.md).
```

## Recent Changes

Add a new section:

```markdown
## Recent Changes

### v1.1.0 - Critical Prediction Fixes

**Major Bug Fixes:**
- Fixed form index / pace model inversion that caused predictions to be near-uniform
- Removed variance-destroying pace scaling factor
- Improved model blending strategy (75/25 model/baseline)
- Better hyperparameters for faster training and generalization

**Added:**
- Comprehensive test suite (optional for contributors)
- MIT License
- Contributing guidelines
- Detailed fix documentation

**Impact:** Predictions now show meaningful hierarchy and correlate strongly with actual results.

For full details, see [FIXES_AND_IMPROVEMENTS.md](FIXES_AND_IMPROVEMENTS.md).
```
