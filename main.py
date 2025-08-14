#!/usr/bin/env python3
"""
F1 predictive app CLI.

Quick start:
  python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  python main.py --season 2025 --round next --sessions qualifying race sprint_qualifying sprint --live --refresh 600

Examples:
  # Predict next round all sessions (one-shot):
  python main.py --round next

  # Predict a specific past round with HTML report and open browser:
  python main.py --season 2021 --round 10 --sessions qualifying race --html --open-browser

  # Live mode with 30s refresh:
  python main.py --season 2025 --round next --live --refresh 30 --html --open-browser

  # Backtest (rolling):
  python main.py --backtest

Flag behaviour:
  - --html controls generating an HTML report. OFF by default.
  - --open-browser implies --html (it will generate HTML and open it).
  - app.open_browser_for_html only controls opening the browser when HTML is already being generated.
"""

import argparse

from f1pred.config import load_config, AppConfig
from f1pred.util import ensure_dirs, get_logger, init_caches
from f1pred.predict import run_predictions_for_event
from f1pred.backtest import run_backtests
from f1pred.live import live_loop
from f1pred.data.fastf1_backend import init_fastf1

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F1 predictive modelling CLI")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--season", type=str, default=None)
    p.add_argument("--round", type=str, default="next")
    p.add_argument(
        "--sessions", nargs="+", default=None,
        help="Session types: qualifying race sprint_qualifying sprint"
    )
    p.add_argument("--html", action="store_true", help="Generate HTML report")
    p.add_argument("--open-browser", action="store_true",
                   help="Open HTML report in browser (implies --html)")
    p.add_argument("--live", action="store_true", help="Enable live mode (periodic refresh)")
    p.add_argument("--refresh", type=int, default=None, help="Live refresh interval in seconds")
    p.add_argument("--backtest", action="store_true", help="Run rolling backtests")
    p.add_argument("--no-cache", action="store_true", help="Disable request caching")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg: AppConfig = load_config(args.config)

    # Apply CLI overrides
    if args.refresh is not None:
        cfg.app.live_refresh_seconds = args.refresh

    # Ensure dirs and install HTTP cache
    ensure_dirs(cfg.paths.output_dir, cfg.paths.reports_dir, cfg.paths.cache_dir, cfg.paths.fastf1_cache)
    init_caches(cfg, disable_cache=args.no_cache)

    # Auto-initialise FastF1 cache if enabled (no-op if not installed)
    try:
        if cfg.data_sources.fastf1.enabled:
            init_fastf1(cfg.paths.fastf1_cache)
    except Exception:
        pass

    if args.backtest:
        run_backtests(cfg)
        return

    sessions = args.sessions or cfg.modelling.targets.session_types

    if args.live:
        live_open_browser = args.open_browser or cfg.app.open_browser_for_html
        live_loop(cfg, season=args.season, rnd=args.round, sessions=sessions,
                  open_browser=live_open_browser)
        return

    # One-shot prediction:
    generate_html = bool(args.html or args.open_browser)
    open_browser_flag = bool(args.open_browser or (cfg.app.open_browser_for_html and generate_html))

    run_predictions_for_event(
        cfg,
        season=args.season,
        rnd=args.round,
        sessions=sessions,
        generate_html=generate_html,
        open_browser=open_browser_flag,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")