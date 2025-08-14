<<<<<<< HEAD
# f1predictor
A tool for predicting the outcome of F1 sessions
=======
F1 Prediction Application

Overview
This application predicts Formula 1 results for upcoming and historical rounds across multiple session types:
- Qualifying
- Race
- Sprint Qualifying (Sprint Shootout)
- Sprint

It uses:
- Jolpica’s Ergast-compatible API for schedules, rosters, and results
- Open-Meteo for historical and forecast weather
- OpenF1 (historical data only) and FastF1 as enrichers for session timing and classification (especially for Sprint Shootout)

Predictions are fully empirical and re-trained on each run (self-calibrating). No model weights are persisted between runs.

Key Features
- Predicts Q, R, SQ, S for past and future rounds
- Time-cut logic: for past predictions, only data available before the event is used
- Automatic roster inference for future rounds (season-aware)
- Recency-weighted driver and team form, teammate head-to-head
- Weather features (temperature, precipitation, pressure, wind, humidity), with configurable units
- DNF risk model for race-like sessions
- Monte Carlo simulation for finish distributions (mean position, win probability, top-3 probability)
- Outputs a tidy CSV and optional HTML report
- Backtesting with metrics and a summary HTML report
- Live mode that re-runs periodically, switches to actuals when posted, and shows movement indicators

Data Sources
- Jolpica F1 (Ergast-compatible): schedules, drivers, constructors, results, standings
- Open-Meteo: historical weather, historical forecast archive, forecast; elevation, geocoding as needed
- OpenF1 (historical only): laps, stints, weather, session metadata (used for Sprint Shootout actuals when Jolpica lacks a distinct endpoint)
- FastF1: event schedule, timings, and classification fallback for Sprint Shootout

Installation
1) Create a virtual environment and install requirements
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # Linux/macOS: source .venv/bin/activate
   pip install -r requirements.txt

2) Confirm Python 3.11+ (recommended). The code uses typing features and scikit-learn versions that match the included requirements.

Configuration
All settings are in config.yaml. Key sections:

- app
  - model_version: arbitrary version string for tracking
  - random_seed: fixed seed for reproducibility
  - live_refresh_seconds: polling interval in live mode
  - open_browser_for_html: default behaviour for opening HTML reports

- paths
  - cache_dir: directory for HTTP cache (requests-cache)
  - fastf1_cache: directory for FastF1 cache
  - output_dir, reports_dir: location for outputs
  - predictions_csv: main CSV output for predictions
  - backtest_metrics_csv: CSV with backtest evaluation metrics
  - backtest_report: HTML summary of backtest metrics

- data_sources
  - jolpica.base_url: must be Ergast-compatible (e.g., https://api.jolpi.ca/ergast/f1)
  - openf1.enabled: true for historical enrichment; no live paid data is used
  - fastf1.enabled: true to use FastF1 for session timing/classification fallback
  - open_meteo: URLs for weather endpoints, plus configurable units
    - temperature_unit: celsius or fahrenheit
    - windspeed_unit: kmh, ms, mph, kn
    - precipitation_unit: mm or inch

- caching
  - requests_cache: HTTP cache settings, with separate TTLs for forecast and historical endpoints

- modelling
  - recency_half_life_days: base, weather, team half-lives for form features
  - monte_carlo.draws: number of simulation draws (default 5000)
  - features: toggles for additional enrichments (tyres/laps if available historically)
  - targets.session_types: which sessions to run by default

- backtesting
  - seasons: auto (last 5 completed seasons), all, or list of specific years
  - metrics: which metrics to compute (spearman, kendall, accuracy_top3, brier_pairwise, crps)

- output
  - colours and toggles for HTML output

Caching and Rate Limits
- HTTP responses are cached via requests-cache. Forecast endpoints get a short TTL (hours), historical endpoints get a longer TTL (days).
- Only GET requests are cached.
- Retries with exponential backoff and Retry-After support are enabled for robustness.

Usage

Basic predictions:
- Predict the next round, all sessions:
  python main.py --round next

- Predict a specific past round and open HTML:
  python main.py --season 2023 --round 5 --sessions qualifying race --html --open-browser

- Predict all sessions and store output only:
  python main.py --season 2024 --round 2

Live mode:
- Live re-run every 30 seconds, with HTML and browser open once:
  python main.py --season 2025 --round next --live --refresh 30 --html --open-browser

Backtesting:
- Run backtests and generate metrics CSV + HTML summary:
  python main.py --backtest

Outputs
- CSV: output/predictions.csv
  Columns:
    season, round, event, driver_id, driver, team,
    predicted_pos, mean_pos, p_top3, p_win, p_dnf,
    actual_pos (if available), delta (actual - predicted),
    generated_at, model_version
  The file is grouped by season/round/session and sorted by predicted_pos.

- HTML report per event: output/reports/{season}_R{round}.html
  Shows predicted ranking with mean position, top-3%, win%, DNF%, and movement when actuals are present.

- Backtest:
  - CSV metrics: output/backtest_metrics.csv (event-level metrics)
  - HTML summary: output/reports/backtest_summary.html (aggregate metrics by session and recent events)

Modelling Details
- Pace model: gradient boosting regressor (LightGBM or XGBoost if available; fall back to sklearn GBDT), trained on features excluding the target to avoid leakage.
- Target: inverse of form_index (lower is better) so model produces a “pace index”.
- DNF model: gradient boosting classifier using driver/team DNF base rates and weather features as a proxy.
- Simulation: Monte Carlo draws add stochasticity and DNF hazards to convert pace to an order. Derives:
  - p_win: probability of winning
  - p_top3: probability of finishing in top 3
  - mean_pos: expected finishing position
  - pairwise probabilities for head-to-head metrics in backtests

Roster Logic
- For future events, the entry list is inferred from the most recent completed round within the same season (race results preferred; qualifying fallback).
- If none exist (e.g., Round 1), season drivers are used and constructors are mapped via driver standings where possible.

Weather Handling
- Aggregates hourly weather around the session window into features: means/min/max for temperature, pressure, wind, gusts, precipitation, etc.
- Configurable units via config.yaml.

Sprint Shootout (Sprint Qualifying) Actuals
- When Jolpica does not supply Sprint Shootout classification, the app:
  - Uses OpenF1 laps to build a classification from best lap per driver_number, or
  - Falls back to FastF1 classification if available (by DriverNumber or Abbreviation).
- This enables delta-to-actuals for Sprint Shootout in predictions.

Reproducibility
- Random seed is fixed via config.yaml.
- The ranking helper’s noise can be made reproducible if wired, but the top-level simulation uses the seed for consistency across runs.

Limitations
- OpenF1 is used only for historical data. No paid real-time OpenF1 endpoints are used.
- Weather sensitivity features are conservative unless richer historical weather alignment is built per event.
- The DNF proxy model uses base rates; a full per-event hazard analysis would be a future enhancement.

Troubleshooting
- If no HTML opens in live mode: pass --html --open-browser explicitly.
- If you see missing actuals for sprint_qualifying, ensure OpenF1 is enabled and/or FastF1 is installed.
- If requests fail due to rate limiting, the built-in retry and cache should mitigate; otherwise increase live_refresh_seconds.

Extending
- Add features to f1pred/features.py and wire into the model in f1pred/models.py.
- Add other weather variables via Open-Meteo by updating the client and aggregate function.
- Integrate additional metrics in f1pred/metrics.py and expose them in the backtest HTML.

Support
If you run into issues with endpoints or data availability, verify the base URLs in config.yaml and re-run with a clean cache (delete the .cache directory).
>>>>>>> af0ba6c (Initial commit)
