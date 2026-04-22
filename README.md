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

- **[FastF1](https://docs.fastf1.dev/)** – Detailed timing and telemetry fallback

## How It Works

1. **Roster inference** – For future races, the entry list comes from the most recent completed event
2. **Feature engineering** – Driver form, team performance, weather conditions, teammate comparisons, starting grid
3. **Grid position handling** – Uses actual grid from race results when available; for pre-race predictions, runs a qualifying simulation to estimate starting positions
4. **Model training** – An ensemble of four specialized models is trained fresh on historical data:
    - **GBM (AI Brain)**: Analyzes patterns like weather and recent momentum to predict raw speed.
    - **Elo (Skill Score)**: A Chess-style rating that tracks a driver's talent relative to their rivals.
    - **BT (Head-to-Head)**: Determines strength by seeing who consistently finishes ahead of whom.
    - **Mix (Talent Separator)**: Mathematically separates a driver's skill from the car's performance.
5. **DNF estimation** – Separate classifier for retirement probability
6. **Monte Carlo simulation** – 5000 draws to get win probability, podium chances, and expected position

### Two-Stage Prediction (Race)

When predicting a race before qualifying has occurred:

1. The system first runs a full qualifying prediction
2. Uses predicted qualifying positions as the starting grid
3. Feeds this grid into the race prediction model

This allows accurate race predictions even before the grid is known, and accounts for grid penalties and unexpected qualifying results when actual data is available.

## Input Variables

The model uses a variety of engineered features to capture driver talent, car performance, and track conditions. These factors are categorized as follows:

### Performance & Form
- **Race Form (`form_index`)**: A recency-weighted score of a driver's finishing positions and points. Higher values indicate better current form.
- **Qualifying Form (`qualifying_form_index`)**: A recency-weighted index of a driver's qualifying performance.
- **Sprint Form (`sprint_form_index`)**: A recency-weighted score of a driver's sprint finishing positions and points.
- **Sprint Quali Form (`sprint_qualifying_form_index`)**: A recency-weighted index of a driver's sprint qualifying performance.
- **Team Strength (`team_form_index`)**: The recent performance level of the constructor, representing the car's competitive ceiling.

### Comparative Metrics
- **vs Teammate (`teammate_delta`)**: The average qualifying advantage over their teammate. A positive value means they consistently out-qualify their peer.
- **Overtaking Skill (`grid_finish_delta`)**: The average number of positions a driver gains (or loses) between their starting grid and final finish, measuring racecraft.

### Track-Specific Factors
- **Circuit History (`circuit_avg_pos`)**: The driver's historical average finishing position at this specific track.
- **Circuit Experience (`circuit_experience`)**: Total number of starts the driver has at this circuit.
- **Circuit DNF Rate (`circuit_dnf_rate`)**: The driver's personal retirement rate at this track.
- **Pass Difficulty (`circuit_overtake_difficulty`)**: A track-wide metric calculating how easy or hard it is to gain positions based on historical data.
- **Track DNF Rate (`global_circuit_dnf_rate`)**: The overall historical retirement rate for all drivers at this circuit.

### Session & Grid
- **Race Session (`is_race`)**: A boolean flag indicating if the session is a race.
- **Quali Session (`is_qualifying`)**: A boolean flag indicating if the session is qualifying.
- **Sprint Session (`is_sprint`)**: A boolean flag indicating if the session is a sprint.
- **Grid Position (`grid`)**: The starting position for the race. This acts as a mathematical "anchor" for race predictions.
- **Quali Position (`current_quali_pos`)**: The actual qualifying result achieved during the current race weekend.

### Weather & Conditions
- **Weather Impact (`weather_effect`)**: The total predicted influence of atmospheric conditions on a driver's performance.
- **Condition Skills (`temp_skill`, `rain_skill`, `wind_skill`, `pressure_skill`)**: Driver-specific proficiency scores for various weather conditions, derived from historical performance correlations.
- **Atmospheric Data (`weather_temp_mean`, `weather_rain_sum`, etc.)**: Raw metrics (temperature, rain, wind, pressure, humidity) for the session window.

## Configuration

All settings live in `config.yaml`. The main things you might want to tweak:

```yaml
modelling:
  # Model weights (including recency, blending, ensemble, DNF)
  # are dynamically calibrated at runtime.
  # Delete calibration_weights.json to force full re-calibration.
  monte_carlo:
    draws: 5000 # Simulation iterations (more = slower but smoother)

data_sources:
  open_meteo:
    temperature_unit: "celsius" # or fahrenheit
    windspeed_unit: "kmh" # kmh, ms, mph, kn
    precipitation_unit: "mm" # mm, inch
```

## Output

**Terminal output and Web UI** – Predictions display directly in the console with:

- Driver names and teams
- Predicted positions
- Win probability, podium probability, DNF probability
- Weather conditions
- Position changes when actual results are available

## Usage Modes

### Web UI

Start the built-in web server to view predictions in a browser:

```bash
python main.py --web --port 8000
```

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

### Testing

Run the local test suite (requires installing test dependencies first):

```bash
make install
pip install pytest pytest-cov httpx playwright fastparquet pyarrow colorama
playwright install --with-deps chromium
make test
```

## Known Limitations

- **Weather is approximate** – Forecasts are aggregated around session windows
- **DNF model is basic** – Uses historical base rates, not detailed reliability analysis
- **First race of season** – Limited data for brand new driver/team combinations
- **Grid penalties** – Only reflected if race results are available (post-qualifying predictions may not account for all penalties)

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
    └── fastf1_backend.py
```

## Requirements

- Python 3.11+
- See `requirements.txt` for dependencies

## Troubleshooting

**Predictions seem random or uniform?**
Clear the cache and re-run:

```bash
rm -rf cache/ .cache/
python main.py --round next
```

**Missing actuals for sprint qualifying?**
Enable FastF1 in `config.yaml`.

**Rate limiting errors?**
The built-in cache and retry logic should handle most cases. Try increasing `live_refresh_seconds` or clearing the `.cache/` directory.

**Import errors for LightGBM on macOS?**

```bash
pip uninstall lightgbm
pip install lightgbm --no-binary lightgbm
```

The system will fall back to XGBoost or scikit-learn if LightGBM is unavailable.

## License

MIT – see [LICENSE](LICENSE)
