# F1 Prediction

![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A Formula 1 race prediction tool that uses historical data and machine learning to forecast qualifying and race results.

> **Note:** This project was built with significant assistance from AI (GitHub Copilot / Claude). The codebase, documentation, and overall architecture were developed collaboratively with AI tools.

## What It Does

This tool predicts finishing positions for F1 sessions:
- **Qualifying** – Grid positions
- **Race** – Final standings
- **Sprint Qualifying** – Sprint grid
- **Sprint** – Sprint race results

It pulls data from public APIs, builds features from historical performance, and trains models fresh on each run—no saved weights, fully self-calibrating.

## Quick Start

```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Predict the next race
python main.py --round next

# Predict a specific past event
python main.py --season 2024 --round 5 --sessions qualifying race
```

## Data Sources

The app fetches data from free, public APIs:

- **[Jolpica F1](https://github.com/jolpica/jolpica-f1)** – Schedules, results, standings (Ergast-compatible)
- **[Open-Meteo](https://open-meteo.com/)** – Weather forecasts and historical weather
- **[OpenF1](https://openf1.org/)** – Session timing data (historical only)
- **[FastF1](https://theoehrly.github.io/Fast-F1/)** – Detailed timing and telemetry fallback

## How It Works

1. **Roster inference** – For future races, the entry list comes from the most recent completed event
2. **Feature engineering** – Driver form, team performance, weather conditions, teammate comparisons
3. **Model training** – Gradient boosting (LightGBM/XGBoost/sklearn) trained on historical data
4. **DNF estimation** – Separate classifier for retirement probability
5. **Monte Carlo simulation** – 5000 draws to get win probability, podium chances, and expected position

## Configuration

All settings live in `config.yaml`. The main things you might want to tweak:

```yaml
modelling:
  recency_half_life_days:
    base: 120      # How quickly old results fade in importance
    weather: 180   # Weather skill memory
    team: 240      # Team performance memory
  monte_carlo:
    draws: 5000    # Simulation iterations (more = slower but smoother)

data_sources:
  open_meteo:
    temperature_unit: "celsius"  # or fahrenheit
    windspeed_unit: "kmh"        # kmh, ms, mph, kn
```

## Output

**Terminal output only** – Predictions display directly in the console with:
- Driver names and teams
- Predicted positions
- Win probability, podium probability, DNF probability
- Weather conditions
- Position changes when actual results are available

## Usage Modes

### Standard Prediction
```bash
python main.py --season 2024 --round 10
```

### Live Mode
Re-runs predictions periodically and updates when results come in:
```bash
python main.py --round next --live --refresh 30
```

### Backtesting
Evaluate model accuracy across historical seasons:
```bash
python main.py --backtest
```

## Known Limitations

- **No real-time data** – OpenF1 is used for historical data only, not live timing
- **Weather is approximate** – Forecasts are aggregated around session windows
- **DNF model is basic** – Uses historical base rates, not detailed reliability analysis
- **First race of season** – Limited data for brand new driver/team combinations

## Project Structure

```
f1pred/
├── predict.py      # Main prediction pipeline
├── features.py     # Feature engineering
├── models.py       # ML model training
├── simulate.py     # Monte Carlo simulation
├── roster.py       # Entry list inference
├── backtest.py     # Historical evaluation
└── data/           # API clients
    ├── jolpica.py
    ├── open_meteo.py
    ├── openf1.py
    └── fastf1_backend.py
```

## Requirements

- Python 3.11+
- See `requirements.txt` for dependencies

## Troubleshooting

**Predictions seem random or uniform?**
Clear the cache and re-run:
```bash
rm -rf .cache/
python main.py --round next
```

**Missing actuals for sprint qualifying?**
Enable OpenF1 and/or install FastF1 in `config.yaml`.

**Rate limiting errors?**
The built-in cache and retry logic should handle most cases. Try increasing `live_refresh_seconds` or clearing the `.cache/` directory.

**Import errors for LightGBM on macOS?**
```bash
pip uninstall lightgbm
pip install lightgbm --no-binary lightgbm
```
The system will fall back to XGBoost or scikit-learn if LightGBM is unavailable.

## Running Tests

If you want to verify the code or contribute:
```bash
pip install pytest
pytest tests/ -v
```

## License

MIT – see [LICENSE](LICENSE)