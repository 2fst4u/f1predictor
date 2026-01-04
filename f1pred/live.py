from __future__ import annotations
from typing import List, Optional
import time

from colorama import Fore, Style

from .util import get_logger
from .data.jolpica import JolpicaClient
from .predict import run_predictions_for_event, resolve_event

logger = get_logger(__name__)

def live_loop(cfg, season: Optional[str], rnd: str, sessions: List[str], open_browser: bool = False) -> None:
    """
    Live mode: periodically refresh predictions.
    """
    jc = JolpicaClient(cfg.data_sources.jolpica.base_url, cfg.data_sources.jolpica.timeout_seconds,
                       cfg.data_sources.jolpica.rate_limit_sleep)
    season_i, round_i, race_info = resolve_event(jc, season, rnd)
    title = f"{race_info.get('raceName')} {season_i} (Round {round_i})"
    refresh = cfg.app.live_refresh_seconds

    # Only open the browser once if requested
    opened_browser_once = False

    logger.info(f"Live mode started for {title} | refresh={refresh}s")
    while True:
        run_predictions_for_event(
            cfg,
            season=str(season_i),
            rnd=str(round_i),
            sessions=sessions,
            generate_html=True,
            open_browser=(open_browser and not opened_browser_once)
        )
        if open_browser and not opened_browser_once:
            opened_browser_once = True

        time.sleep(refresh)