"""Main prediction pipeline for F1 events.

This module orchestrates feature building, model training, simulation, and
output generation for F1 race predictions.
"""
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
from .data.fastf1_backend import init_fastf1, get_session_classification, get_session_weather_status
from .features import build_session_features, collect_historical_results
from .models import train_pace_model, estimate_dnf_probabilities
from .simulate import simulate_grid
from .ensemble import EloModel, BradleyTerryModel, MixedEffectsLikeModel, EnsembleConfig, combine_pace

__all__ = [
    "run_predictions_for_event",
    "resolve_event",
]

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
                        except Exception as e:
                            logger.debug(f"Could not get race results for {s} R{race['round']}: {e}")
                            continue
                    else:
                        r = races[-1]["round"]
                else:
                    now = datetime.now(timezone.utc)
                    future = [x for x in races if datetime.fromisoformat(x["date"] + "T00:00:00+00:00") >= now]
                    r = future[0]["round"] if future else races[-1]["round"]
            else:
                r = rnd
        race_info = [x for x in jc.get_season_schedule(str(s)) if str(x.get("round")) == str(r)]
        if not race_info:
            logger.info(f"[predict] Could not resolve schedule for {s} round {r}; continuing with defaults")
            race_info = [{
                "raceName": None,
                "date": datetime.now(timezone.utc).date().isoformat(),
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
                "date": datetime.now(timezone.utc).date().isoformat(),
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
    return_results: bool = False,
):
    """Generate predictions for given event with terminal output only.

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
    date_str = race_info.get("date") or datetime.now(timezone.utc).date().isoformat()
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
            pace_model, pace_hat, feat_cols = train_pace_model(X, session_type=sess, cfg=cfg)

            # Standardize GBM pace (z-score) but preserve variance
            # Note: We do NOT apply pace_scale compression here - that destroys signal
            try:
                mu = float(np.mean(pace_hat))
                sd = float(np.std(pace_hat))
                if not np.isfinite(sd) or sd < 1e-6:
                    logger.warning("[predict] Pace predictions have very low variance (std=%.6f)", sd)
                    sd = 1.0
                pace_hat = (pace_hat - mu) / sd
                # Log variance for debugging
                logger.info("[predict] GBM pace standardized: mean=%.4f, std=%.4f", mu, sd)
            except Exception as e:
                logger.warning("[predict] Pace standardization failed: %s; using raw GBM pace", e)

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
                        cfg=cfg
                    )
                except Exception as e:
                    logger.info(f"[predict] DNF estimation failed; using default 0.12: {e}")
                    dnf_prob[:] = 0.12

            # Monte Carlo simulation
            draws = cfg.modelling.monte_carlo.draws
            prob_matrix, mean_pos, pairwise = simulate_grid(
                combined_pace, 
                dnf_prob, 
                draws=draws,
                noise_factor=cfg.modelling.simulation.noise_factor,
                min_noise=cfg.modelling.simulation.min_noise,
                max_penalty_base=cfg.modelling.simulation.max_penalty_base
            )

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
                        "code": row.get("code"),
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

            # Check for wet session via FastF1
            is_wet = False
            try:
                session_name_map = {
                    "race": "Race",
                    "qualifying": "Qualifying", 
                    "sprint": "Sprint",
                    "sprint_qualifying": "Sprint Shootout",
                }
                ff1_sess_name = session_name_map.get(sess, sess.title())
                weather_status = get_session_weather_status(season_i, round_i, ff1_sess_name)
                if weather_status:
                    is_wet = weather_status.get("is_wet", False)
            except Exception:
                pass
            
            print_session_console(ranked, sess, cfg, meta.get("weather"), is_wet=is_wet, event_date=ref_date)

        except Exception as e:
                logger.info(f"[predict] Session {sess} failed with exception:")
                logger.info(f"{type(e).__name__}: {e}")
                import traceback
                logger.info(traceback.format_exc())
                continue

    if return_results:
        return {
            "season": season_i,
            "round": round_i,
            "sessions": session_results,
        }


def print_session_console(df: pd.DataFrame, sess: str, cfg, weather_info: Optional[Dict[str, float]] = None, is_wet: bool = False, event_date: Optional[datetime] = None) -> None:
    title = _session_title(sess)
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}== {title} =={Style.RESET_ALL}")
    
    # Weather display with condition-based coloring
    if weather_info:
        t = weather_info.get("temp_mean")
        r = weather_info.get("rain_sum")
        w = weather_info.get("wind_mean")
        
        # Check if weather data is valid (not NaN)
        import math
        has_valid_weather = (
            t is not None and not math.isnan(t) and
            r is not None and not math.isnan(r)
        )
        
        if has_valid_weather:
            w_parts = []
            # Color temp: blue if cold (<15°C), red if hot (>30°C), white otherwise
            if t < 15:
                temp_color = Fore.BLUE
            elif t > 30:
                temp_color = Fore.RED
            else:
                temp_color = Fore.WHITE
            w_parts.append(f"{temp_color}{t:.0f}°C{Style.RESET_ALL}")
            
            # Color rain: cyan if any rain
            rain_color = Fore.CYAN if r > 0 else Fore.WHITE
            w_parts.append(f"{rain_color}Rain:{r:.1f}mm{Style.RESET_ALL}")
            
            if w is not None and not math.isnan(w):
                # Color wind: yellow if strong (>20km/h)
                wind_color = Fore.YELLOW if w > 20 else Fore.WHITE
                w_parts.append(f"{wind_color}Wind:{w:.0f}km/h{Style.RESET_ALL}")
            
            # Add wet indicator
            if is_wet:
                w_parts.append(f"{Fore.CYAN}{Style.BRIGHT}[WET]{Style.RESET_ALL}")
            
            print(f"Weather: {' | '.join(w_parts)}")
        else:
            # Weather data unavailable - show when it will be available
            if event_date:
                now = datetime.now(timezone.utc)
                days_until = (event_date - now).days
                forecast_available_in = max(0, days_until - 7)  # Forecasts typically 7 days ahead
                if forecast_available_in > 0:
                    print(f"{Style.DIM}Weather: Unknown (forecast available in ~{forecast_available_in} days){Style.RESET_ALL}")
                else:
                    print(f"{Style.DIM}Weather: Unknown{Style.RESET_ALL}")
            else:
                print(f"{Style.DIM}Weather: Unknown{Style.RESET_ALL}")
    
    # Calculate column widths for alignment
    max_name = max(len((r.get("name") or "")[:22]) for _, r in df.iterrows()) if not df.empty else 18
    max_team = max(len((r.get("constructorName") or "")[:14]) for _, r in df.iterrows()) if not df.empty else 10
    max_name = max(max_name, 14)  # Minimum for readability
    max_team = max(max_team, 10)  # Minimum for readability
    
    # Session-specific column labels
    is_quali = sess in ("qualifying", "sprint_qualifying")
    win_label = "Pole" if is_quali else "Win"
    
    # Print column headers
    header = (
        f"{Style.DIM}{'#':>3}   "
        f"{'Driver':<{max_name}}   "
        f"{'Team':<{max_team+2}}   "
        f"{'Avg':>5}   "
        f"{'Top3':>6}   "
        f"{win_label:>6}   "
        f"{'DNF':>6}   "
        f"{'Pos':>3}{Style.RESET_ALL}"
    )
    print(header)
    # Horizontal separator
    sep_width = 3 + 3 + max_name + 3 + max_team + 2 + 3 + 5 + 3 + 6 + 3 + 6 + 3 + 6 + 3 + 3
    print(f"{Style.DIM}{'─' * sep_width}{Style.RESET_ALL}")
    
    for _, r in df.iterrows():
        pos = int(r["predicted_position"])
        name = (r.get("name") or "")[:max_name]
        team = (r.get("constructorName") or "")[:max_team]
        mp = float(r["mean_pos"])
        top3 = float(r["p_top3"]) * 100
        win = float(r["p_win"]) * 100
        dnf = float(r["p_dnf"]) * 100
        
        # Color coding for probabilities
        win_color = Fore.GREEN if win > 25 else Fore.WHITE
        dnf_color = Fore.RED if dnf > 15 else Fore.WHITE
        top3_color = Fore.GREEN if top3 > 75 else Fore.WHITE
        
        # Actual classification display
        actual_pos = r.get("actual_position")
        if pd.notna(actual_pos):
            classified_str = f"{Fore.CYAN}{Style.BRIGHT}{int(actual_pos):>3d}{Style.RESET_ALL}"
        else:
            classified_str = f"{Style.DIM}{'--':>3}{Style.RESET_ALL}"
        
        # Left-aligned columns with good padding
        print(
            f"{Fore.YELLOW}{pos:>3}.{Style.RESET_ALL}  "
            f"{Fore.CYAN}{name:<{max_name}}{Style.RESET_ALL}   "
            f"{Style.DIM}[{team:<{max_team}}]{Style.RESET_ALL}   "
            f"{mp:5.1f}   "
            f"{top3_color}{top3:5.1f}%{Style.RESET_ALL}   "
            f"{win_color}{win:5.1f}%{Style.RESET_ALL}   "
            f"{dnf_color}{dnf:5.1f}%{Style.RESET_ALL}   "
            f"{classified_str}"
        )

