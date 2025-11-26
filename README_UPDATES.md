# README Updates - Add These Sections

Add these sections to your README.md:

## Badges

Add at the top of README.md after the title:

```markdown
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)
```

## Development Setup

Add after Installation section:

```markdown
## Development Setup

For contributors and developers:

1. Clone the repository:
   ```bash
   git clone https://github.com/2fst4u/f1predictor.git
   cd f1predictor
   ```

2. Set up development environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   make install-dev
   ```

3. Run tests:
   ```bash
   make test
   ```

4. Lint and format code:
   ```bash
   make lint
   make format
   ```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.
```

## Troubleshooting

Add before the Extending section:

```markdown
## Troubleshooting

### Predictions seem random or uniform

**Solution:** Clear the cache and re-run:
```bash
rm -rf .cache/
python main.py --round next
```

The cache may contain outdated feature calculations from before the fixes.

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

### Tests failing

Ensure you have dev dependencies:
```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

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

**Infrastructure:**
- Added comprehensive test suite with pytest
- Code quality tools (ruff, mypy)
- Pre-commit hooks for local development
- Makefile for common commands

**Documentation:**
- Added LICENSE (MIT)
- Added CONTRIBUTING.md
- Added FIXES_AND_IMPROVEMENTS.md with detailed fix explanations

**Impact:** Predictions now show meaningful hierarchy and correlate strongly with actual results.

For full details, see [FIXES_AND_IMPROVEMENTS.md](FIXES_AND_IMPROVEMENTS.md).
```
