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
import random
import difflib
from datetime import datetime

from f1pred.config import load_config, AppConfig
from f1pred.util import ensure_dirs, get_logger, init_caches, configure_logging, sanitize_for_console
from f1pred.predict import run_predictions_for_event
from f1pred.backtest import run_backtests
from f1pred.live import live_loop
from f1pred.data.fastf1_backend import init_fastf1

logger = get_logger(__name__)


def _print_random_tip() -> None:
    """Display a helpful tip for the user to discover features."""
    tips = [
        "Use --live to auto-refresh predictions during the race.",
        "Validate model accuracy with --backtest.",
        "Check specific sessions with --sessions qualifying race.",
        "Use --log-level debug for deeper insights into the prediction model.",
        "Specify --season and --round to replay historical events.",
        "See --help for a full list of commands and examples.",
    ]
    tip = random.choice(tips)
    print(f"\n{Style.DIM}💡 Tip: {tip}{Style.RESET_ALL}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="F1 predictive modelling CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
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
        current_year = datetime.now().year
        if not args.season.isdigit() or len(args.season) != 4:
            # Sentinel: Sanitize user input before printing to console
            safe_season = sanitize_for_console(args.season)
            print(f"{Fore.RED}✖ Invalid season '{safe_season}'.{Style.RESET_ALL} Please use 'current' or a 4-digit year (e.g. {current_year}).")
            return

        season_year = int(args.season)
        if season_year < 1950 or season_year > current_year + 1:
            safe_season = sanitize_for_console(args.season)
            print(f"{Fore.RED}✖ Invalid season '{safe_season}'.{Style.RESET_ALL} F1 data is available from 1950 to {current_year + 1}.")
            return

    if args.round and args.round not in ("next", "last"):
        if not args.round.isdigit():
            # Sentinel: Sanitize user input before printing to console
            safe_round = sanitize_for_console(args.round)
            print(f"{Fore.RED}✖ Invalid round '{safe_round}'.{Style.RESET_ALL} Please use 'next', 'last', or a round number.")
            return

    if args.sessions:
        valid_sessions = set(cfg.modelling.targets.session_types)
        # 🎨 Palette: Map common aliases to canonical names
        aliases = {
            "q": "qualifying", "qual": "qualifying", "quali": "qualifying",
            "r": "race", "gp": "race",
            "s": "sprint",
            "sq": "sprint_qualifying", "shootout": "sprint_qualifying", "sprint_quali": "sprint_qualifying"
        }

        normalized_sessions = []
        for s in args.sessions:
            s_lower = s.lower()
            if s_lower in valid_sessions:
                normalized_sessions.append(s_lower)
                continue

            if s_lower in aliases:
                normalized_sessions.append(aliases[s_lower])
                continue

            # Invalid - try to find suggestion
            # Sentinel: Sanitize user input before printing to console
            safe_s = sanitize_for_console(s)
            print(f"{Fore.RED}✖ Invalid session '{safe_s}'.{Style.RESET_ALL}")

            suggestions = difflib.get_close_matches(s_lower, valid_sessions, n=1, cutoff=0.6)
            if suggestions:
                print(f"  Did you mean '{suggestions[0]}'?")

            print(f"  Allowed types: {', '.join(sorted(valid_sessions))}")
            return

        args.sessions = normalized_sessions

    # Apply CLI overrides
    if args.refresh is not None:
        if args.refresh < 1:
            print(f"{Fore.RED}✖ Invalid refresh interval.{Style.RESET_ALL} Must be at least 1 second.")
            return
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

    _print_random_tip()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}✖ Operation cancelled by user.{Style.RESET_ALL}")
    except Exception as e:
        # Catch-all for cleaner error output
        # Sentinel: Sanitize exception message to prevent log injection/terminal spoofing
        safe_msg = sanitize_for_console(str(e))
        print(f"\n{Fore.RED}✖ An unexpected error occurred:{Style.RESET_ALL} {safe_msg}")
        print(f"{Style.DIM}Run with --log-level debug for more details.{Style.RESET_ALL}")
        sys.exit(1)