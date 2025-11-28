from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
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
from .models import train_pace_model, train_dnf_hazard_model, estimate_dnf_probabilities
from .simulate import simulate_grid
from .report import generate_html_report
from .ensemble import EloModel, BradleyTerryModel, MixedEffectsLikeModel, EnsembleConfig, combine_pace

logger = get_logger(__name__)


def resolve_event(jc: JolpicaClient, season: Optional[str], rnd: str) -> Tuple[int, int, Dict[str, Any]]:
    """Resolve season/round to use. Never raises; falls back to current/last if needed."""
    try:
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
                        try:
                            if jc.get_race_results(str(s), race["round"]):
                                r = race["round"]
                                break
                        except Exception:
                            continue
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
            logger.info(f"[predict] Could not resolve schedule for {s} round {r}; continuing with defaults")
            race_info = [{
                "raceName": None,
                "date": datetime.utcnow().date().isoformat(),
                "time": "00:00:00+00:00",
            }]
        return int(s), int(r), race_info[0]
    except Exception as e:
        logger.info(f"[predict] resolve_event failed: {e}; falling back to current/last")
        s, r = jc.get_latest_season_and_round()
        fallback_info = [x for x in jc.get_season_schedule(str(s)) if str(x.get("round")) == str(r)]
        if not fallback_info:
            fallback_info = [{
                "raceName": None,
                "date": datetime.utcnow().date().isoformat(),
                "time": "00:00:00+00:00",
            }]
        return int(s), int(r), fallback_info[0]


def _session_title(stype: str) -> str:
    return {
        "race": "Grand Prix (Race)",
        "qualifying": "Qualifying",
        "sprint": "Sprint",
        "sprint_qualifying": "Sprint Qualifying",
    }.get(stype, stype)


def _parse_lap_seconds(v) -> float:
    """Accept numeric seconds or strings "M:SS.mmm" / "SS.mmm"."""
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
    season_i: int,
    round_i: int,
    sess: str,
    roster_view: pd.DataFrame,  # expects columns: driverId, number, code
) -> Optional[pd.Series]:
    """Return a Series aligned to roster_view with actual finishing/qualifying position where available.

    Uses Jolpica for race/qual/sprint. For sprint_qualifying, uses FastF1 classification if present.
    Never raises; returns None if not available.
    """
    try:
        if sess == "race":
            act = jc.get_race_results(str(season_i), str(round_i))
            amap = {r["Driver"]["driverId"]: int(r["position"]) for r in act if r.get("position")}
            return roster_view["driverId"].map(amap) if amap else None

        if sess == "qualifying":
            act = jc.get_qualifying_results(str(season_i), str(round_i))
            amap = {r["Driver"]["driverId"]: int(r["position"]) for r in act if r.get("position")}
            return roster_view["driverId"].map(amap) if amap else None

        if sess == "sprint":
            act = jc.get_sprint_results(str(season_i), str(round_i))
            amap = {r["Driver"]["driverId"]: int(r["position"]) for r in act if r.get("position")}
            return roster_view["driverId"].map(amap) if amap else None

        if sess == "sprint_qualifying":
            cls = get_session_classification(season_i, round_i, "Sprint Shootout")
            if cls is not None and hasattr(cls, "empty") and not cls.empty:
                if "DriverNumber" in cls.columns:
                    num_series = pd.to_numeric(roster_view["number"], errors="coerce").astype("Int64")
                    num_to_pos = dict(
                        cls.dropna(subset=["Position"]).astype({"DriverNumber": int, "Position": int})[
                            ["DriverNumber", "Position"]
                        ].values
                    )
                    return num_series.map(num_to_pos)
                if "Abbreviation" in cls.columns:
                    code_series = roster_view["code"].astype(str)
                    abbr_to_pos = dict(
                        cls.dropna(subset=["Position"])[["Abbreviation", "Position"]].values
                    )
                    return code_series.map(abbr_to_pos)
            return None

        return None
    except Exception:
        return None


def _filter_sessions_for_round(jc: JolpicaClient, season_i: int, round_i: int, requested: List[str]) -> List[str]:
    """Assume a regular weekend unless confirmed otherwise.

    - Always keep qualifying and race.
    - Include sprint and sprint_qualifying only if the current round already has sprint results posted.
    """
    keep = []
    requested_norm = [s.strip().lower() for s in requested]

    for s in ("qualifying", "race"):
        if s in requested_norm:
            keep.append(s)

    try:
        has_sprint = bool(jc.get_sprint_results(str(season_i), str(round_i)))
    except Exception:
        has_sprint = False

    if has_sprint:
        for s in ("sprint_qualifying", "sprint"):
            if s in requested_norm:
                keep.append(s)

    return keep


def run_predictions_for_event(
    cfg,
    season: Optional[str],
    rnd: str,
    sessions: List[str],
    generate_html: bool = True,
    open_browser: bool = False,
    return_results: bool = False,
):
    """Generate predictions for given event, write CSV and optional HTML.

    Returns a dict when return_results=True for the backtester.
    Never raises on normal control flow; logs and skips sessions if necessary.
    """
    # Ensure FastF1 cache is initialised (no-op if disabled/not installed)
    try:
        if cfg.data_sources.fastf1.enabled:
            ensure_dirs(cfg.paths.fastf1_cache)
            init_fastf1(cfg.paths.fastf1_cache)
    except Exception:
        pass

    # API clients
    jc = JolpicaClient(
        cfg.data_sources.jolpica.base_url,
        cfg.data_sources.jolpica.timeout_seconds,
        cfg.data_sources.jolpica.rate_limit_sleep,
    )
    om = OpenMeteoClient(
        cfg.data_sources.open_meteo.forecast_url,
        cfg.data_sources.open_meteo.historical_weather_url,
        cfg.data_sources.open_meteo.historical_forecast_url,
        cfg.data_sources.open_meteo.elevation_url,
        cfg.data_sources.open_meteo.geocoding_url,
        timeout=cfg.data_sources.jolpica.timeout_seconds,
        temperature_unit=cfg.data_sources.open_meteo.temperature_unit,
        windspeed_unit=cfg.data_sources.open_meteo.windspeed_unit,
        precipitation_unit=cfg.data_sources.open_meteo.precipitation_unit,
    )
    of1 = OpenF1Client(
        cfg.data_sources.openf1.base_url,
        cfg.data_sources.openf1.timeout_seconds,
        cfg.data_sources.openf1.enabled,
    )

    # Resolve event
    season_i, round_i, race_info = resolve_event(jc, season, rnd)
    event_title = f"{race_info.get('raceName') or 'Event'} {season_i} (Round {round_i})"

    # Event datetime (UTC)
    date_str = race_info.get("date") or datetime.utcnow().date().isoformat()
    time_str = race_info.get("time")
    if time_str:
        if time_str.endswith("Z"):
            timespec = time_str.replace("Z", "+00:00")
        else:
            timespec = f"{time_str}+00:00"
    else:
        timespec = "00:00:00+00:00"
    try:
        event_date = datetime.fromisoformat(f"{date_str}T{timespec}")
        if event_date.tzinfo is None:
            event_date = event_date.replace(tzinfo=timezone.utc)
    except Exception:
        event_date = datetime.now(timezone.utc)

    # Filter requested sessions for this round
    sessions = _filter_sessions_for_round(jc, season_i, round_i, sessions)

    all_preds: List[Dict[str, Any]] = []
    session_results: Dict[str, Dict[str, Any]] = {}

    for sess in sessions:
        try:
            logger.info(f"Building features for {event_title} - {sess}")
            ref_date = event_date

            X, meta, roster = build_session_features(jc, om, of1, season_i, round_i, sess, ref_date, cfg)
            if (
                X is None
                or roster is None
                or (hasattr(X, "empty") and X.empty)
                or (hasattr(roster, "empty") and roster.empty)
            ):
                logger.info(f"[predict] No features/roster available for {sess}; skipping")
                continue

            # Train pace model
            pace_model, pace_hat, feat_cols = train_pace_model(X, session_type=sess)

            # Normalise GBM pace for stability
            try:
                mu = float(np.mean(pace_hat))
                sd = float(np.std(pace_hat))
                if not np.isfinite(sd) or sd < 1e-6:
                    sd = 1.0
                pace_hat = (pace_hat - mu) / sd
                pace_scale = getattr(getattr(cfg, "modelling", object()), "pace_scale", 0.6)
                try:
                    pace_scale = float(pace_scale)
                except Exception:
                    pace_scale = 0.6
                pace_hat = pace_hat * pace_scale
            except Exception:
                logger.info("[predict] Pace normalisation failed; using raw GBM pace")

            # Historical results for this roster
            roster_ids = roster["driverId"].dropna().astype(str).tolist() if not roster.empty else []
            hist = collect_historical_results(
                jc,
                season=season_i,
                end_before=ref_date,
                lookback_years=75,
                roster_driver_ids=roster_ids,
            )

            # --- Ensemble skill components (all data-driven) ---
            elo_pace = bt_pace = mixed_pace = None
            try:
                elo_model = EloModel().fit(hist)
                elo_pace = elo_model.predict(X)
            except Exception as e:
                logger.info(f"[predict] Elo model failed: {e}")

            try:
                bt_model = BradleyTerryModel().fit(hist)
                bt_pace = bt_model.predict(X)
            except Exception as e:
                logger.info(f"[predict] Bradley–Terry model failed: {e}")

            try:
                mixed_model = MixedEffectsLikeModel().fit(hist)
                mixed_pace = mixed_model.predict(X)
            except Exception as e:
                logger.info(f"[predict] Mixed-effects-like model failed: {e}")

            # Combine GBM pace with ensemble elements
            try:
                ens_cfg = EnsembleConfig()  # could later be fed from cfg.modelling.ensemble
                combined_pace = combine_pace(
                    gbm_pace=pace_hat,
                    elo_pace=elo_pace,
                    bt_pace=bt_pace,
                    mixed_pace=mixed_pace,
                    cfg=ens_cfg,
                )
                logger.info(
                    "[predict] Combined pace stats: std=%.4f, range=%.4f",
                    float(np.std(combined_pace)),
                    float(np.ptp(combined_pace)),
                )
            except Exception as e:
                logger.info(f"[predict] Ensemble combine failed, falling back to GBM pace: {e}")
                combined_pace = pace_hat

            # DNF probabilities
            dnf_prob = np.zeros(X.shape[0], dtype=float)
            if sess in ("race", "sprint"):
                try:
                    dnf_prob = estimate_dnf_probabilities(
                        hist,
                        X,
                        alpha=2.0,
                        beta=8.0,
                        driver_weight=0.6,
                        team_weight=0.4,
                        clip_min=0.02,
                        clip_max=0.35,
                    )
                except Exception as e:
                    logger.info(f"[predict] DNF estimation failed; using default 0.12: {e}")
                    dnf_prob[:] = 0.12

            # Monte Carlo simulation
            draws = cfg.modelling.monte_carlo.draws
            prob_matrix, mean_pos, pairwise = simulate_grid(combined_pace, dnf_prob, draws=draws)

            p_top3 = prob_matrix[:, :3].sum(axis=1)
            p_win = prob_matrix[:, 0]
            order = np.argsort(mean_pos)
            ranked = X.iloc[order].reset_index(drop=True)
            ranked["mean_pos"] = mean_pos[order]
            ranked["p_top3"] = p_top3[order]
            ranked["p_win"] = p_win[order]
            ranked["p_dnf"] = dnf_prob[order]
            ranked["predicted_position"] = np.arange(1, len(ranked) + 1)

            # Ensure required columns exist for actuals mapping
            if "number" not in ranked.columns:
                ranked["number"] = pd.Series([pd.NA] * len(ranked))
            if "code" not in ranked.columns:
                ranked["code"] = ""

            actual_positions = _get_actual_positions_for_session(
                jc,
                season_i,
                round_i,
                sess,
                ranked[["driverId", "number", "code"]],
            )
            if actual_positions is not None:
                ranked["actual_position"] = actual_positions
                ranked["delta"] = ranked["actual_position"] - ranked["predicted_position"]
            else:
                ranked["actual_position"] = np.nan
                ranked["delta"] = np.nan

            for _, row in ranked.iterrows():
                all_preds.append(
                    {
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
                        "model_version": cfg.app.model_version,
                    }
                )

            print_session_console(ranked, sess, cfg)

        except Exception as e:
            logger.info(f"[predict] Session {sess} failed: {e}; skipping")
            continue

    # Write CSV
    try:
        outcsv = cfg.paths.predictions_csv
        ensure_dirs(os.path.dirname(outcsv))
        newdf = pd.DataFrame(all_preds)
        merged = newdf
        if os.path.exists(outcsv) and os.path.getsize(outcsv) > 0:
            try:
                old = pd.read_csv(outcsv)
                merged = pd.concat(
                    [
                        old[
                            ~(
                                (old.season == season_i)
                                & (old.round == round_i)
                                & (old.event.isin(sessions))
                            )
                        ],
                        newdf,
                    ],
                    ignore_index=True,
                )
            except Exception:
                # corrupted or empty-parse file; overwrite with newdf
                merged = newdf
        if not merged.empty:
            merged = merged.sort_values(["season", "round", "event", "predicted_pos"])
        merged.to_csv(outcsv, index=False)
        logger.info(f"Predictions written to {outcsv}")
    except Exception as e:
        logger.info(f"[predict] Writing predictions CSV failed: {e}")

    # HTML report
    try:
        if generate_html:
            report_path = os.path.join(cfg.paths.reports_dir, f"{season_i}_R{round_i}.html")
            subtitle = f"{event_title}"
            generate_html_report(
                report_path,
                title=event_title,
                subtitle=subtitle,
                sessions_data=[],
                cfg=cfg,
                open_browser=open_browser,
            )
    except Exception as e:
        logger.info(f"[predict] HTML report generation failed: {e}")

    if return_results:
        return {
            "season": season_i,
            "round": round_i,
            "sessions": session_results,
        }


def print_session_console(df: pd.DataFrame, sess: str, cfg) -> None:
    title = _session_title(sess)
    print(f"\n== {title} ==")
    for _, r in df.iterrows():
        pos = int(r["predicted_position"])
        name = (r.get("name") or "")[:30]
        team = (r.get("constructorName") or "")[:18]
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
        print(
            f"{pos:2d}. {name:30s} [{team:18s}]  μ={mp:4.1f}  "
            f"Top3={top3:4.1f}%  Win={win:4.1f}%  DNF={dnf:4.1f}%  {delta_str}"
        )
