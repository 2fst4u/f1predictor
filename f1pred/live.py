from __future__ import annotations
from typing import List, Optional
import time
import os

import pandas as pd
from colorama import Fore, Style

from .util import get_logger
from .data.jolpica import JolpicaClient
from .predict import run_predictions_for_event, resolve_event

logger = get_logger(__name__)

def live_loop(cfg, season: Optional[str], rnd: str, sessions: List[str], open_browser: bool = False) -> None:
    """
    Live mode: periodically refresh predictions and switch to actuals when available.
    Shows change indicators for drivers whose predicted position has improved or worsened vs previous loop.
    """
    jc = JolpicaClient(cfg.data_sources.jolpica.base_url, cfg.data_sources.jolpica.timeout_seconds,
                       cfg.data_sources.jolpica.rate_limit_sleep)
    season_i, round_i, race_info = resolve_event(jc, season, rnd)
    title = f"{race_info.get('raceName')} {season_i} (Round {round_i})"
    refresh = cfg.app.live_refresh_seconds

    last_positions = {sess: None for sess in sessions}
    last_actuals = {sess: None for sess in sessions}

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

        csv_path = cfg.paths.predictions_csv
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df[(df["season"] == season_i) & (df["round"] == round_i) & (df["event"].isin(sessions))]

            for sess in sessions:
                dfs = df[df["event"] == sess].copy().sort_values("predicted_pos")
                cur_positions = list(dfs["driver_id"])
                cur_actuals_available = dfs["actual_pos"].notna().any()

                if last_positions[sess] is not None:
                    prev_order = last_positions[sess]
                    movement = {}
                    for i, drv in enumerate(cur_positions):
                        if drv in prev_order:
                            prev_idx = prev_order.index(drv)
                            delta = prev_idx - i
                            movement[drv] = delta
                    print(f"\nΔ changes since last refresh ({sess}):")
                    for i, drv in enumerate(cur_positions):
                        d = movement.get(drv, 0)
                        name = dfs.iloc[i]["driver"]
                        if d > 0:
                            print(Fore.GREEN + f"↑ {name} (+{d})" + Style.RESET_ALL)
                        elif d < 0:
                            print(Fore.RED + f"↓ {name} ({d})" + Style.RESET_ALL)
                        else:
                            print(f"· {name} (0)")
                last_positions[sess] = cur_positions

                if cur_actuals_available and not last_actuals.get(sess, False):
                    print(f"\nOfficial results posted for {sess}. Differences vs prediction:")
                    for _, row in dfs.iterrows():
                        name = row["driver"]
                        pred = int(row["predicted_pos"])
                        act = int(row["actual_pos"]) if pd.notna(row["actual_pos"]) else None
                        if act is None:
                            continue
                        diff = act - pred
                        if diff < 0:
                            diff_str = Fore.GREEN + f"↑ {abs(diff)}" + Style.RESET_ALL
                        elif diff > 0:
                            diff_str = Fore.RED + f"↓ {diff}" + Style.RESET_ALL
                        else:
                            diff_str = "· 0"
                        print(f"{name:20s} predicted {pred:2d} → actual {act:2d} | {diff_str}")
                    last_actuals[sess] = True

        time.sleep(refresh)