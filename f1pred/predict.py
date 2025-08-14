from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os

import numpy as np
import pandas as pd
from colorama import Fore, Style

from .util import get_logger, ensure_dirs
from .data.jolpica import JolpicaClient
from .data.open_meteo import OpenMeteoClient
from .data.openf1 import OpenF1Client
from .data.fastf1_backend import init_fastf1, get_session_classification
from .features import build_session_features, collect_historical_results
from .models import train_pace_model, train_dnf_hazard_model
from .simulate import simulate_grid
from .report import generate_html_report

logger = get_logger(__name__)


def resolve_event(jc: JolpicaClient, season: Optional[str], rnd: str) -> Tuple[int, int, Dict[str, Any]]:
    if season is None or (isinstance(season, str) and season.lower() == "current"):
        if rnd == "next":
            s, r = jc.get_next_round()
        elif rnd == "last":
            s, r = jc.get_latest_season_and_round()
        else:
            s, r = jc.get_latest_season_and_round()
            r = rnd
    else:
        s = season
        if rnd in ("next", "last"):
            races = jc.get_season_schedule(str(s))
            if rnd == "last":
                for race in reversed(races):
                    if jc.get_race_results(str(s), race["round"]):
                        r = race["round"]
                        break
                else:
                    r = races[-1]["round"]
            else:
                now = datetime.utcnow()
                future = [x for x in races if datetime.fromisoformat(x["date"] + "T00:00:00+00:00") >= now]
                r = future[0]["round"] if future else races[-1]["round"]
        else:
            r = rnd
    race_info = [x for x in jc.get_season_schedule(str(s)) if str(x.get("round")) == str(r)]
    if not race_info:
        raise ValueError(f"Could not resolve schedule for {s} round {r}")
    return int(s), int(r), race_info[0]


def _session_title(stype: str) -> str:
    return {
        "race": "Grand Prix (Race)",
        "qualifying": "Qualifying",
        "sprint": "Sprint",
        "sprint_qualifying": "Sprint Qualifying"
    }.get(stype, stype)


def _parse_lap_seconds(v) -> float:
    # Accept numeric seconds or strings "M:SS.mmm" / "SS.mmm"
    if v is None:
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return np.nan
    try:
        if ":" in s:
            m, rest = s.split(":", 1)
            return float(m) * 60.0 + float(rest)
        return float(s)
    except Exception:
        return np.nan


def _get_actual_positions_for_session(
    jc: JolpicaClient,
    of1: OpenF1Client,
    season_i: int,
    round_i: int,
    sess: str,
    roster_view: pd.DataFrame  # expects columns: driverId, number, code
) -> Optional[pd.Series]:
    """
    Return a Series aligned to roster_view with actual position where available.
    Supports:
      - race, qualifying, sprint via Jolpica
      - sprint_qualifying via OpenF1 laps classification (best lap per driver_number) with FastF1 fallback
    """
    try:
        if sess == "race":
            act = jc.get_race_results(str(season_i), str(round_i))
            amap = {r["Driver"]["driverId"]: int(r["position"]) for r in act if r.get("position")}
            return roster_view["driverId"].map(amap)

        if sess == "qualifying":
            act = jc.get_qualifying_results(str(season_i), str(round_i))
            amap = {r["Driver"]["driverId"]: int(r["position"]) for r in act if r.get("position")}
            return roster_view["driverId"].map(amap)

        if sess == "sprint":
            act = jc.get_sprint_results(str(season_i), str(round_i))
            amap = {r["Driver"]["driverId"]: int(r["position"]) for r in act if r.get("position")}
            return roster_view["driverId"].map(amap)

        if sess == "sprint_qualifying":
            # 1) Try OpenF1: find Sprint Shootout, compute best lap per driver_number
            if of1 and of1.enabled:
                skey = of1.find_session(season_i, round_i, "Sprint Shootout")
                if skey:
                    laps = of1.get_laps(skey)
                    if not laps.empty and "driver_number" in laps.columns:
                        time_col = None
                        for cand in ("lap_duration", "lap_time", "duration"):
                            if cand in laps.columns:
                                time_col = cand
                                break
                        if time_col is not None:
                            laps["_lap_sec"] = laps[time_col].apply(_parse_lap_seconds)
                            grp = laps.groupby("driver_number")["_lap_sec"].min().reset_index().rename(
                                columns={"_lap_sec": "best_lap_seconds"}
                            )
                            grp["position"] = grp["best_lap_seconds"].rank(method="min", ascending=True, na_option="bottom").astype("Int64")
                            # Map permanent numbers to positions
                            num_series = pd.to_numeric(roster_view["number"], errors="coerce").astype("Int64")
                            num_to_pos = dict(grp.dropna(subset=["position"]).astype({"driver_number": int, "position": int})[["driver_number", "position"]].values)
                            return num_series.map(num_to_pos)

            # 2) FastF1 fallback: use classification (DriverNumber or Abbreviation)
            ff_cls = get_session_classification(season_i, round_i, "Sprint Shootout")
            if ff_cls is not None and not ff_cls.empty:
                if "DriverNumber" in ff_cls.columns:
                    num_series = pd.to_numeric(roster_view["number"], errors="coerce").astype("Int64")
                    num_to_pos = dict(ff_cls.dropna(subset=["Position"]).astype({"DriverNumber": int, "Position": int})[["DriverNumber", "Position"]].values)
                    return num_series.map(num_to_pos)
                if "Abbreviation" in ff_cls.columns:
                    code_series = roster_view["code"].astype(str)
                    abbr_to_pos = dict(ff_cls.dropna(subset=["Position"])[["Abbreviation", "Position"]].values)
                    return code_series.map(abbr_to_pos)

            return None

        return None
    except Exception:
        return None


def run_predictions_for_event(cfg, season: Optional[str], rnd: str, sessions: List[str],
                              generate_html: bool = True, open_browser: bool = False,
                              return_results: bool = False):
    """
    Generate predictions for given event, write CSV and optional HTML.
    If return_results=True, also return a dict of per-session outputs for metric computation:
      { session: {"ranked": DataFrame, "prob_matrix": np.ndarray, "pairwise": np.ndarray} }
    """
    # Ensure FastF1 cache is initialised (no-op if disabled/not installed)
    try:
        if cfg.data_sources.fastf1.enabled:
            ensure_dirs(cfg.paths.fastf1_cache)
            init_fastf1(cfg.paths.fastf1_cache)
    except Exception:
        pass

    jc = JolpicaClient(cfg.data_sources.jolpica.base_url, cfg.data_sources.jolpica.timeout_seconds,
                       cfg.data_sources.jolpica.rate_limit_sleep)
    om = OpenMeteoClient(
        cfg.data_sources.open_meteo.forecast_url,
        cfg.data_sources.open_meteo.historical_weather_url,
        cfg.data_sources.open_meteo.historical_forecast_url,
        cfg.data_sources.open_meteo.elevation_url,
        cfg.data_sources.open_meteo.geocoding_url,
        timeout=cfg.data_sources.jolpica.timeout_seconds,
        temperature_unit=cfg.data_sources.open_meteo.temperature_unit,
        windspeed_unit=cfg.data_sources.open_meteo.windspeed_unit,
        precipitation_unit=cfg.data_sources.open_meteo.precipitation_unit
    )
    of1 = OpenF1Client(cfg.data_sources.openf1.base_url, cfg.data_sources.openf1.timeout_seconds, cfg.data_sources.openf1.enabled)

    season_i, round_i, race_info = resolve_event(jc, season, rnd)
    event_title = f"{race_info.get('raceName')} {season_i} (Round {round_i})"
    event_date = datetime.fromisoformat(race_info.get("date") + "T" + race_info.get("time", "00:00:00").replace("Z", "+00:00"))

    sessions_out = []
    all_preds = []
    session_results: Dict[str, Dict[str, Any]] = {}

    for sess in sessions:
        logger.info(f"Building features for {event_title} - {sess}")
        ref_date = event_date
        X, meta, roster = build_session_features(jc, om, of1, season_i, round_i, sess, ref_date, cfg)
        if X.empty:
            logger.warning(f"No features available for session {sess}; skipping")
            continue

        pace_model, pace_hat, feat_cols = train_pace_model(X, session_type=sess)

        hist = collect_historical_results(jc, season=season_i, end_before=ref_date, lookback_years=75)
        dnf_model = None
        dnf_prob = np.zeros(X.shape[0], dtype=float)
        if sess in ("race", "sprint"):
            dnf_model = train_dnf_hazard_model(X, hist)
            if dnf_model is not None:
                clf, cols = dnf_model
                Xdnf = X.copy()
                for c in cols:
                    if c not in Xdnf.columns:
                        Xdnf[c] = 0.0
                dnf_prob = clf.predict_proba(Xdnf[cols])[:, 1]
            else:
                dnf_prob[:] = 0.1

        draws = cfg.modelling.monte_carlo.draws
        prob_matrix, mean_pos, pairwise = simulate_grid(pace_hat, dnf_prob, draws=draws)

        p_top3 = prob_matrix[:, :3].sum(axis=1)
        p_win = prob_matrix[:, 0]
        order = np.argsort(mean_pos)
        ranked = X.iloc[order].reset_index(drop=True)
        ranked["mean_pos"] = mean_pos[order]
        ranked["p_top3"] = p_top3[order]
        ranked["p_win"] = p_win[order]
        ranked["p_dnf"] = dnf_prob[order]
        ranked["predicted_position"] = np.arange(1, len(ranked) + 1)

        # Actuals (including Sprint Shootout via helper)
        actual_positions = _get_actual_positions_for_session(jc, of1, season_i, round_i, sess, ranked[["driverId", "number", "code"]])
        if actual_positions is not None:
            ranked["actual_position"] = actual_positions
            ranked["delta"] = ranked["actual_position"] - ranked["predicted_position"]
        else:
            ranked["actual_position"] = np.nan
            ranked["delta"] = np.nan

        for _, row in ranked.iterrows():
            all_preds.append({
                "season": season_i,
                "round": round_i,
                "event": sess,
                "driver_id": row["driverId"],
                "driver": row.get("name"),
                "team": row.get("constructorName"),
                "predicted_pos": int(row["predicted_position"]),
                "mean_pos": float(row["mean_pos"]),
                "p_top3": float(row["p_top3"]),
                "p_win": float(row["p_win"]),
                "p_dnf": float(row["p_dnf"]),
                "actual_pos": int(row["actual_position"]) if pd.notna(row["actual_position"]) else None,
                "delta": int(row["delta"]) if pd.notna(row["delta"]) else None,
                "generated_at": pd.Timestamp.utcnow().isoformat(),
                "model_version": cfg.app.model_version
            })

        print_session_console(ranked, sess, cfg)

        sess_rows = []
        for _, row in ranked.iterrows():
            sess_rows.append({
                "rank": int(row["predicted_position"]),
                "name": row["name"],
                "code": row.get("code") or "",
                "team": row.get("constructorName") or "",
                "mean_pos": float(row["mean_pos"]),
                "p_top3": float(row["p_top3"]),
                "p_win": float(row["p_win"]),
                "p_dnf": float(row["p_dnf"]),
                "delta": (int(row["delta"]) if pd.notna(row["delta"]) else None)
            })
        sessions_out.append({"session_title": _session_title(sess), "rows": sess_rows})

        prob_ord = prob_matrix[order]
        pairwise_ord = pairwise[order][:, order]
        session_results[sess] = {
            "ranked": ranked.copy(),
            "prob_matrix": prob_ord,
            "pairwise": pairwise_ord,
        }

    outcsv = cfg.paths.predictions_csv
    ensure_dirs(os.path.dirname(outcsv))
    newdf = pd.DataFrame(all_preds)
    if os.path.exists(outcsv):
        old = pd.read_csv(outcsv)
        merged = pd.concat([old[~((old.season == season_i) & (old.round == round_i) & (old.event.isin(sessions)))], newdf],
                           ignore_index=True)
    else:
        merged = newdf
    merged = merged.sort_values(["season", "round", "event", "predicted_pos"])
    merged.to_csv(outcsv, index=False)
    logger.info(f"Predictions written to {outcsv}")

    if generate_html:
        report_path = os.path.join(cfg.paths.reports_dir, f"{season_i}_R{round_i}.html")
        subtitle = f"{event_title}"
        generate_html_report(report_path, title=event_title, subtitle=subtitle, sessions_data=sessions_out,
                             cfg=cfg, open_browser=open_browser)

    if return_results:
        return {
            "season": season_i,
            "round": round_i,
            "event_title": event_title,
            "sessions": session_results
        }


def print_session_console(df: pd.DataFrame, sess: str, cfg) -> None:
    title = _session_title(sess)
    print(f"\n== {title} ==")
    for _, r in df.iterrows():
        pos = int(r["predicted_position"])
        name = r.get("name")
        team = r.get("constructorName") or ""
        mp = float(r["mean_pos"])
        top3 = float(r["p_top3"]) * 100
        win = float(r["p_win"]) * 100
        dnf = float(r["p_dnf"]) * 100
        delta = r.get("delta")
        if pd.notna(delta):
            if delta < 0:
                delta_str = Fore.GREEN + f"↑ {-int(delta)}" + Style.RESET_ALL
            elif delta > 0:
                delta_str = Fore.RED + f"↓ {int(delta)}" + Style.RESET_ALL
            else:
                delta_str = "·"
        else:
            delta_str = "·"
        print(f"{pos:2d}. {name:20s} [{team:15s}]  μ={mp:4.1f}  Top3={top3:4.1f}%  Win={win:4.1f}%  DNF={dnf:4.1f}%  {delta_str}")