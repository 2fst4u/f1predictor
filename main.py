#!/usr/bin/env python3
r"""
F1 predictive app CLI.

Quick start:
  python -m venv .venv && . .venv/bin/activate  # Windows: .venv\\Scripts\\activate
  pip install -r requirements.txt
  python main.py --season 2025 --round next --sessions qualifying race sprint_qualifying sprint --live --refresh 600

Examples:
  # Predict next round all sessions (one-shot):
  python main.py --round next

  # Predict a specific past round:
  python main.py --season 2021 --round 10 --sessions qualifying race

  # Live mode with 30s refresh:
  python main.py --season 2025 --round next --live --refresh 30

  # Backtest (rolling):
  python main.py --backtest
"""

import argparse

from colorama import Fore, Style
import sys

from f1pred.config import load_config, AppConfig
from f1pred.util import ensure_dirs, get_logger, init_caches, configure_logging
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
    p.add_argument("--live", action="store_true", help="Enable live mode (periodic refresh)")
    p.add_argument("--refresh", type=int, default=None, help="Live refresh interval in seconds")
    p.add_argument("--backtest", action="store_true", help="Run rolling backtests")
    p.add_argument("--no-cache", action="store_true", help="Disable request caching")
    p.add_argument(
        "--log-level", type=str, default=None,
        choices=["debug", "info", "warning", "error"],
        help="Set logging verbosity (default: from config, or WARNING)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg: AppConfig = load_config(args.config)

    # Configure logging - CLI overrides config
    log_level = args.log_level.upper() if args.log_level else cfg.app.log_level
    configure_logging(log_level)
    
    # Re-get logger after configuration
    global logger
    logger = get_logger(__name__)

    # Input validation (UX Improvement)
    # Fail fast with a friendly message instead of falling back to default/current
    if args.season and args.season != "current":
        if not args.season.isdigit() or len(args.season) != 4:
            print(f"{Fore.RED}✖ Invalid season '{args.season}'.{Style.RESET_ALL} Please use 'current' or a 4-digit year (e.g. 2025).")
            return

    if args.round and args.round not in ("next", "last"):
        if not args.round.isdigit():
            print(f"{Fore.RED}✖ Invalid round '{args.round}'.{Style.RESET_ALL} Please use 'next', 'last', or a round number.")
            return

    # Apply CLI overrides
    if args.refresh is not None:
        cfg.app.live_refresh_seconds = args.refresh

    # Ensure dirs and install HTTP cache
    ensure_dirs(cfg.paths.cache_dir, cfg.paths.fastf1_cache)
    init_caches(cfg, disable_cache=args.no_cache)

    # Auto-initialise FastF1 cache if enabled (no-op if not installed)
    try:
        if cfg.data_sources.fastf1.enabled:
            init_fastf1(cfg.paths.fastf1_cache)
    except Exception as e:
        logger.warning(f"FastF1 initialization failed (non-fatal): {e}")

    if args.backtest:
        run_backtests(cfg)
        return

    sessions = args.sessions or cfg.modelling.targets.session_types

    if args.live:
        live_loop(cfg, season=args.season, rnd=args.round, sessions=sessions)
        return

    # One-shot prediction:
    run_predictions_for_event(
        cfg,
        season=args.season,
        rnd=args.round,
        sessions=sessions,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}✖ Operation cancelled by user.{Style.RESET_ALL}")
    except Exception as e:
        # Catch-all for cleaner error output
        print(f"\n{Fore.RED}✖ An unexpected error occurred:{Style.RESET_ALL} {e}")
        print(f"{Style.DIM}Run with --log-level debug for more details.{Style.RESET_ALL}")
        sys.exit(1)