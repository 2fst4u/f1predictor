from __future__ import annotations
from typing import List, Dict, Any
from datetime import datetime, timezone
import os

import pandas as pd

from .util import get_logger, ensure_dirs
from .data.jolpica import JolpicaClient
from .predict import run_predictions_for_event
from .metrics import compute_event_metrics
from .report import generate_backtest_summary_html

logger = get_logger(__name__)

def _auto_backtest_seasons(jc: JolpicaClient) -> List[int]:
    try:
        js = jc.get_season_schedule("current")
        current_year = int(js[0]["season"])
        years = list(range(current_year - 5, current_year))
        return years
    except Exception:
        return [2020, 2021, 2022, 2023]

def run_backtests(cfg) -> None:
    jc = JolpicaClient(cfg.data_sources.jolpica.base_url, cfg.data_sources.jolpica.timeout_seconds,
                       cfg.data_sources.jolpica.rate_limit_sleep)
    seasons = cfg.backtesting.seasons
    if seasons == "auto":
        season_list = _auto_backtest_seasons(jc)
    elif seasons == "all":
        season_list = list(range(1950, datetime.now(timezone.utc).year))
    elif isinstance(seasons, list):
        season_list = [int(x) for x in seasons]
    else:
        season_list = _auto_backtest_seasons(jc)

    metrics_rows: List[Dict[str, Any]] = []

    for season in season_list:
        races = jc.get_season_schedule(str(season))
        for r in races:
            rnd = r["round"]
            logger.info(f"Backtest: {season} R{rnd} {r['raceName']}")
            try:
                res = run_predictions_for_event(cfg, season=str(season), rnd=str(rnd),
                                                sessions=cfg.modelling.targets.session_types,
                                                generate_html=False, open_browser=False, return_results=True)
                if not res:
                    continue
                for sess, sdata in res["sessions"].items():
                    ranked = sdata["ranked"]
                    prob_matrix = sdata["prob_matrix"]
                    pairwise = sdata["pairwise"]
                    mrow = compute_event_metrics(ranked, prob_matrix, pairwise, session=sess,
                                                 season=res["season"], rnd=res["round"])
                    metrics_rows.append(mrow)

                    outcsv = cfg.paths.backtest_metrics_csv
                    ensure_dirs(os.path.dirname(outcsv))
                    pd.DataFrame(metrics_rows).to_csv(outcsv, index=False)

            except Exception as e:
                logger.warning(f"Backtest prediction failed for {season} R{rnd}: {e}")
                continue

    outcsv = cfg.paths.backtest_metrics_csv
    if os.path.exists(outcsv):
        generate_backtest_summary_html(outcsv, cfg.paths.backtest_report, cfg)
        logger.info(f"Backtesting complete. Metrics CSV: {outcsv} | HTML: {cfg.paths.backtest_report}")
    else:
        logger.info("Backtesting complete, but no metrics were produced.")