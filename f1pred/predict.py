"""Main prediction pipeline for F1 events.

This module orchestrates feature building, model training, simulation, and
output generation for F1 race predictions.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import os

import numpy as np
import pandas as pd
from colorama import Fore, Style

from .util import get_logger, ensure_dirs, StatusSpinner, sanitize_for_console
from .data.jolpica import JolpicaClient
from .data.open_meteo import OpenMeteoClient
from .data.fastf1_backend import init_fastf1, get_session_classification, get_session_weather_status
from .features import build_session_features, collect_historical_results
from .models import train_pace_model, estimate_dnf_probabilities
from .simulate import simulate_grid
from .ensemble import EloModel, BradleyTerryModel, MixedEffectsLikeModel, EnsembleConfig, combine_pace
from .ranking import plackett_luce_scores

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


def _get_session_datetime(race_info: Dict[str, Any], sess: str) -> Optional[datetime]:
    """Extract session datetime from race_info dictionary."""
    if sess == "race":
        d, t = race_info.get("date"), race_info.get("time")
    else:
        # Map to Ergast keys
        key = {
            "practice_1": "FirstPractice", "practice_2": "SecondPractice",
            "practice_3": "ThirdPractice", "qualifying": "Qualifying",
            "sprint": "Sprint", "sprint_qualifying": "SprintQualifying"
        }.get(sess)
        if not key or key not in race_info: return None

        # Ensure session data is a dict (API might return null/None)
        s_data = race_info[key]
        if not isinstance(s_data, dict): return None
        d, t = s_data.get("date"), s_data.get("time")

    if not d: return None
    t = t or "00:00:00Z"
    # Ensure timezone offset exists
    if not t.endswith("Z") and "+" not in t and "-" not in t[-5:]: t += "+00:00"
    try:
        dt = datetime.fromisoformat(f"{d}T{t.replace('Z', '+00:00')}")
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except ValueError:
        return None


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


def _run_single_prediction(
    jc: JolpicaClient,
    om: OpenMeteoClient,
    season_i: int,
    round_i: int,
    sess: str,
    ref_date: datetime,
    cfg,
    X_override: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """Run prediction for a single session and return ranked DataFrame.
    
    If X_override is provided, use it instead of building features.
    Returns None if prediction cannot be completed.
    """
    from .features import build_session_features, collect_historical_results
    
    # Build features (or use override)
    hist = None
    if X_override is not None:
        X = X_override.copy()
        roster = X_override.copy()
        meta = {}
    else:
        X, meta, roster, hist = build_session_features(jc, om, season_i, round_i, sess, ref_date, cfg)
    
    if X is None or roster is None or X.empty or roster.empty:
        return None
    
    # Train pace model
    pace_model, pace_hat, feat_cols = train_pace_model(X, session_type=sess, cfg=cfg)
    
    # Standardize pace
    try:
        mu = float(np.mean(pace_hat))
        sd = float(np.std(pace_hat))
        if not np.isfinite(sd) or sd < 1e-6:
            sd = 1.0
        pace_hat = (pace_hat - mu) / sd
    except Exception:
        pass
    
    # Historical results for ensemble models
    if hist is None:
        roster_ids = roster["driverId"].dropna().astype(str).tolist() if not roster.empty else []
        hist = collect_historical_results(
            jc,
            season=season_i,
            end_before=ref_date,
            lookback_years=75,
            roster_driver_ids=roster_ids,
        )
    
    # Ensemble skill components
    elo_pace = bt_pace = mixed_pace = None
    try:
        elo_model = EloModel().fit(hist)
        elo_pace = elo_model.predict(X)
    except Exception:
        pass
    try:
        bt_model = BradleyTerryModel().fit(hist)
        bt_pace = bt_model.predict(X)
    except Exception:
        pass
    try:
        mixed_model = MixedEffectsLikeModel().fit(hist)
        mixed_pace = mixed_model.predict(X)
    except Exception:
        pass
    
    # Combine pace
    try:
        ens_cfg = EnsembleConfig()
        combined_pace = combine_pace(
            gbm_pace=pace_hat,
            elo_pace=elo_pace,
            bt_pace=bt_pace,
            mixed_pace=mixed_pace,
            cfg=ens_cfg,
        )
    except Exception:
        combined_pace = pace_hat
    
    # DNF probabilities (only for race/sprint)
    dnf_prob = np.zeros(X.shape[0], dtype=float)
    if sess in ("race", "sprint"):
        try:
            dnf_prob = estimate_dnf_probabilities(hist, X, cfg=cfg, event_weather=meta.get("weather"))
        except Exception:
            dnf_prob[:] = 0.12
    
    # Monte Carlo simulation
    draws = cfg.modelling.monte_carlo.draws
    prob_matrix, mean_pos, pairwise = simulate_grid(
        combined_pace,
        dnf_prob,
        draws=draws,
        noise_factor=cfg.modelling.simulation.noise_factor,
        min_noise=cfg.modelling.simulation.min_noise,
        max_penalty_base=cfg.modelling.simulation.max_penalty_base,
        compute_pairwise=False,
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
    
    return ranked


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

    # Resolve event
    season_i, round_i, race_info = resolve_event(jc, season, rnd)
    event_title = f"{race_info.get('raceName') or 'Event'} {season_i} (Round {round_i})"

    # Extract circuit info if available
    circuit_name = None
    if "Circuit" in race_info and isinstance(race_info["Circuit"], dict):
        circuit_name = race_info["Circuit"].get("circuitName")

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
    
    # Sort sessions chronologically to ensure history flows correctly
    # Standard order: Practice -> Sprint Shootout -> Sprint -> Qualifying -> Race
    session_order_map = {
        "practice_1": 1, 
        "practice_2": 2, 
        "practice_3": 3,
        "sprint_qualifying": 4, 
        "sprint": 5, 
        "qualifying": 6, 
        "race": 7,
    }
    sessions.sort(key=lambda s: session_order_map.get(s, 99))
    
    all_preds: List[Dict[str, Any]] = []
    session_results: Dict[str, Dict[str, Any]] = {}

    # Accumulate results from sessions within this run to feed into subsequent sessions
    accumulated_history: List[Dict[str, Any]] = []

    # Cache for ensemble models to avoid re-fitting on identical history/roster
    ensemble_cache: Dict[Tuple[str, ...], Tuple[Optional[EloModel], Optional[BradleyTerryModel], Optional[MixedEffectsLikeModel]]] = {}

    # Cache roster across sessions for the same event to avoid re-deriving it
    cached_roster: Optional[pd.DataFrame] = None

    for sess in sessions:
        try:
            # Resolve specific session datetime if available, otherwise fallback to race event date
            sess_dt = _get_session_datetime(race_info, sess)
            ref_date = sess_dt if sess_dt else event_date
            
            # Wrap heavy operations in a spinner
            with StatusSpinner(f"Predicting {event_title} - {sess}...") as spinner:
                # Convert accumulator to DataFrame for injection
                extra_hist_df = pd.DataFrame(accumulated_history) if accumulated_history else None

                spinner.update(f"Predicting {event_title} - {sess}: Building features...")
                X, meta, roster, hist = build_session_features(
                    jc, om, season_i, round_i, sess, ref_date, cfg,
                    extra_history=extra_hist_df,
                    roster_override=cached_roster
                )

                if cached_roster is None and roster is not None and not roster.empty:
                    cached_roster = roster

                if (
                    X is None
                    or roster is None
                    or (hasattr(X, "empty") and X.empty)
                    or (hasattr(roster, "empty") and roster.empty)
                ):
                    msg = f"Skipping {sess} (no data)"
                    spinner.update(msg)
                    spinner.set_status("skipped")
                    logger.info(f"[predict] {msg}")
                    continue

                # Universal Grid Feature logic (Race<-Quali, Sprint<-SprintQuali)
                # Map target session -> precursor session that determines grid
                grid_precursor_map = {
                    "race": "qualifying",
                    "sprint": "sprint_qualifying",
                }

                has_grid_concept = sess in grid_precursor_map

                grid_source = "actual"
                if has_grid_concept and "grid" in X.columns:
                    if X["grid"].isna().any():
                        precursor = grid_precursor_map[sess]
                        logger.info(f"[predict] Grid not available for {sess} - looking for {precursor} results")
                        
                        # 1. Check if precursor was already run in this loop (internal consistency)
                        precursor_results = [p for p in accumulated_history if p["session"] == precursor]
                        
                        if precursor_results:
                            logger.info(f"[predict] Using {precursor} results from current run as grid")
                            # precursor_results is list of dicts: need map driverId -> position
                            # 'position' in accumulated_history typically maps to the finish position
                            # 'accumulated_history' rows should have 'driverId' and 'position'

                            grid_map = {r["driverId"]: int(r["position"]) for r in precursor_results}
                            X["grid"] = X["driverId"].map(grid_map)
                            grid_source = f"predicted (from {precursor} in loop)"
                            
                        else:
                            # 2. Run simulation if not in loop
                            logger.info(f"[predict] {precursor} not in current loop - running simulation to estimate grid")
                            spinner.update(f"Predicting {event_title} - {sess}: Simulating {precursor} for grid...")
                            # Note: _run_single_prediction does NOT see accumulated_history currently,
                            # but it's a cold start anyway if not in loop.
                            qual_ranked = _run_single_prediction(
                                jc, om, season_i, round_i, precursor, ref_date, cfg
                            )
                            if qual_ranked is not None and not qual_ranked.empty:
                                grid_map = dict(zip(
                                    qual_ranked["driverId"],
                                    qual_ranked["predicted_position"]
                                ))
                                X["grid"] = X["driverId"].map(grid_map)
                                grid_source = f"predicted (from simulated {precursor})"

                                # Optional: append this simulation to history?
                                # Maybe complex to convert format. Rely on simple map for now.
                            else:
                                # 3. Fallback
                                if "form_index" in X.columns:
                                    X["grid"] = X["form_index"].rank(ascending=False, method="first").astype(int)
                                    grid_source = "estimated (from form index)"
                                else:
                                    X["grid"] = np.arange(1, len(X) + 1)
                                    grid_source = "default (no data)"
                    else:
                        logger.info(f"[predict] Using actual grid for {sess}")

                # Train pace model
                spinner.update(f"Predicting {event_title} - {sess}: Training pace model...")
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

                # --- Ensemble skill components (all data-driven) ---
                spinner.update(f"Predicting {event_title} - {sess}: Running ensemble models...")
                elo_pace = bt_pace = mixed_pace = None
                elo_model = bt_model = mixed_model = None

                roster_key = tuple(sorted(roster_ids))
                if roster_key in ensemble_cache:
                    logger.debug(f"[predict] Using cached ensemble models for roster size {len(roster_ids)}")
                    elo_model, bt_model, mixed_model = ensemble_cache[roster_key]
                else:
                    try:
                        elo_model = EloModel().fit(hist)
                    except Exception as e:
                        logger.info(f"[predict] Elo model fit failed: {e}")

                    try:
                        bt_model = BradleyTerryModel().fit(hist)
                    except Exception as e:
                        logger.info(f"[predict] Bradley–Terry model fit failed: {e}")

                    try:
                        mixed_model = MixedEffectsLikeModel().fit(hist)
                    except Exception as e:
                        logger.info(f"[predict] Mixed-effects-like model fit failed: {e}")

                    ensemble_cache[roster_key] = (elo_model, bt_model, mixed_model)

                # Predict using models (new X each session)
                if elo_model:
                    try:
                        elo_pace = elo_model.predict(X)
                    except Exception as e:
                        logger.info(f"[predict] Elo predict failed: {e}")

                if bt_model:
                    try:
                        bt_pace = bt_model.predict(X)
                    except Exception as e:
                        logger.info(f"[predict] BT predict failed: {e}")

                if mixed_model:
                    try:
                        mixed_pace = mixed_model.predict(X)
                    except Exception as e:
                        logger.info(f"[predict] Mixed predict failed: {e}")

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
                            cfg=cfg,
                            event_weather=meta.get("weather"),
                        )
                    except Exception as e:
                        logger.info(f"[predict] DNF estimation failed; using default 0.12: {e}")
                        dnf_prob[:] = 0.12

                # Monte Carlo simulation
                spinner.update(f"Predicting {event_title} - {sess}: Simulating Monte Carlo...")
                draws = cfg.modelling.monte_carlo.draws
                prob_matrix, mean_pos, pairwise = simulate_grid(
                    combined_pace,
                    dnf_prob,
                    draws=draws,
                    noise_factor=cfg.modelling.simulation.noise_factor,
                    min_noise=cfg.modelling.simulation.min_noise,
                    max_penalty_base=cfg.modelling.simulation.max_penalty_base,
                    compute_pairwise=return_results,
                )

                # Analytical probabilities (Plackett-Luce)
                # combined_pace is lower-is-better, so we negate it for higher-is-better scores
                analytical_p_win = plackett_luce_scores(-combined_pace, temperature=1.0)

            p_top3 = prob_matrix[:, :3].sum(axis=1)
            sim_p_win = prob_matrix[:, 0]

            # Blend simulation win prob with analytical win prob (50/50)
            # This makes the final probability more robust to simulation variance
            p_win = 0.5 * sim_p_win + 0.5 * analytical_p_win

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

            # Store structured results for backtesting/return
            session_results[sess] = {
                "ranked": ranked,
                "prob_matrix": prob_matrix,
                "pairwise": pairwise,
                "meta": meta,
            }

            for _, row in ranked.iterrows():
                # Add to flat list for reporting/backtesting
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
                
                # Add to accumulated history for subsequent sessions in this run
                # We need columns: driverId, position, date, session, constructorId, points, qpos, grid
                # Note: 'points' are estimates, 'qpos' is relevant for 'qualifying' session
                accumulated_history.append({
                    "driverId": row["driverId"],
                    "position": int(row["predicted_position"]),
                    "date": ref_date,
                    "session": sess,
                    "constructorId": row.get("constructorId"),
                    "points": 0.0, # Placeholder
                    "grid": row.get("grid"),
                    "qpos": int(row["predicted_position"]) if sess in ("qualifying", "sprint_qualifying") else np.nan
                })

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
            
            print_session_console(
                ranked,
                sess,
                cfg,
                meta.get("weather"),
                is_wet=is_wet,
                event_date=ref_date,
                session_date=sess_dt,
                circuit_name=circuit_name
            )

        except Exception as e:
                logger.info(f"[predict] Session {sess} failed with exception:")
                # Sentinel: Sanitize exception message to prevent log injection
                logger.info(f"{type(e).__name__}: {sanitize_for_console(str(e))}")
                import traceback
                logger.debug(traceback.format_exc())
                continue

    if return_results:
        return {
            "season": season_i,
            "round": round_i,
            "sessions": session_results,
        }


def _render_bar_parts(percentage: float, width: int = 5) -> Tuple[str, str]:
    """Render a high-resolution progress bar using block characters.

    Returns (filled_part, empty_part).
    filled_part contains solid blocks and the partial block.
    empty_part contains filler characters.
    """
    blocks = " ▏▎▍▌▋▊▉█"
    full_value = (percentage / 100.0) * width
    full_chars = int(full_value)
    remainder = full_value - full_chars

    # Select partial block (8 levels)
    partial_idx = int(remainder * 8)
    partial_char = blocks[partial_idx] if partial_idx > 0 else ""

    # Build the filled part
    filled = "█" * full_chars
    if full_chars < width:
        filled += partial_char

    # Calculate empty space length
    current_len = len(filled)
    remaining_len = width - current_len

    empty = "░" * remaining_len

    return filled, empty


def _get_prob_color(value: float, is_dnf: bool = False) -> str:
    """Return color code based on probability value (0-100)."""
    if is_dnf:
        # Lower is better for DNF
        if value < 5: return Fore.GREEN
        if value < 15: return Fore.YELLOW
        return Fore.RED

    # Higher is better for Win/Top3
    if value < 10: return Style.DIM
    if value < 40: return Fore.CYAN
    if value < 70: return Fore.GREEN
    return Fore.YELLOW


def _render_actual_pos(predicted: int, actual: int, width: int = 6) -> str:
    """Render actual position with accuracy-based color coding and alignment."""
    diff = abs(predicted - actual)

    # Determine color and symbol
    if diff == 0:
        color = Fore.GREEN
        symbol = "✓"
    elif diff <= 2:
        color = Fore.CYAN
        symbol = "≈"
    elif diff <= 5:
        color = Fore.YELLOW
        symbol = ""
    else:
        # For very poor predictions, use Red to highlight the discrepancy
        color = Fore.RED
        symbol = ""

    num_str = str(actual)

    # Calculate padding for right alignment
    # Note: We assume '✓' takes 1 char width in the terminal.
    content_len = len(num_str) + len(symbol)
    padding = max(0, width - content_len)
    pad_str = " " * padding

    return f"{pad_str}{color}{Style.BRIGHT}{symbol}{num_str}{Style.RESET_ALL}"


def print_session_console(
    df: pd.DataFrame,
    sess: str,
    cfg,
    weather_info: Optional[Dict[str, float]] = None,
    is_wet: bool = False,
    event_date: Optional[datetime] = None,
    session_date: Optional[datetime] = None,
    circuit_name: Optional[str] = None
) -> None:
    title = _session_title(sess)
    header_line = f"\n{Fore.YELLOW}{Style.BRIGHT}== {title}"
    if circuit_name:
        header_line += f" | {sanitize_for_console(circuit_name)}"
    header_line += f" =={Style.RESET_ALL}"
    print(header_line)
    
    # Display session time if available
    if session_date:
        now = datetime.now(timezone.utc)
        diff = session_date - now
        total_seconds = int(diff.total_seconds())

        # Format nice date: "Sat 14:00 UTC"
        date_str = session_date.strftime("%a %H:%M UTC")

        # Add Local Time (UX Improvement)
        try:
            local_dt = session_date.astimezone()
            # Check if local timezone is effectively different from UTC display
            if local_dt.utcoffset() != timedelta(0):
                tz_name = local_dt.strftime("%Z").strip()
                if not tz_name:
                    # Fallback to offset if name is empty
                    tz_name = local_dt.strftime("%z").strip()

                local_str = f"{local_dt.strftime('%H:%M')} {tz_name}"
                date_str += f" ({local_str})"
        except Exception:
            pass

        if total_seconds > 0:
            days = total_seconds // 86400
            hours = (total_seconds % 86400) // 3600
            minutes = (total_seconds % 3600) // 60

            parts = []
            if days > 0:
                parts.append(f"{days}d")
            if hours > 0:
                parts.append(f"{hours}h")
            if minutes > 0:
                parts.append(f"{minutes}m")

            if not parts:
                rel_str = "<1m"
            else:
                rel_str = " ".join(parts)
            print(f"{Style.DIM}🕒 {date_str} (Starts in {rel_str}){Style.RESET_ALL}")
        elif total_seconds > -10800: # Within 3 hours after start
            print(f"{Style.DIM}🕒 {date_str} (In Progress / Just Finished){Style.RESET_ALL}")
        else:
            print(f"{Style.DIM}🕒 {date_str} (Finished){Style.RESET_ALL}")

    # Weather display with condition-based coloring
    if weather_info:
        t = weather_info.get("temp_mean")
        r = weather_info.get("rain_sum")
        w = weather_info.get("wind_mean")
        h = weather_info.get("humidity_mean")
        c = weather_info.get("cloud_mean")
        
        # Check if weather data is valid (not NaN)
        import math
        has_valid_weather = (
            t is not None and not math.isnan(t) and
            r is not None and not math.isnan(r)
        )
        
        if has_valid_weather:
            w_parts = []

            # 1. Cloud Cover (Sky Condition)
            if c is not None and not math.isnan(c):
                if c < 20:
                    w_parts.append(f"{Style.BRIGHT}☀️{Style.RESET_ALL}")
                elif c < 60:
                    w_parts.append(f"{Style.BRIGHT}🌤️{Style.RESET_ALL}")
                else:
                    w_parts.append(f"{Style.DIM}☁️{Style.RESET_ALL}")

            # 2. Temperature
            # Color temp: cyan if cold (<15°C) for better readability, red if hot (>30°C), white otherwise
            if t < 15:
                temp_color = Fore.CYAN
            elif t > 30:
                temp_color = Fore.RED
            else:
                temp_color = Fore.RESET
            w_parts.append(f"{temp_color}🌡️ {t:.0f}°C{Style.RESET_ALL}")
            
            # 3. Humidity (New)
            if h is not None and not math.isnan(h):
                w_parts.append(f"{Fore.BLUE}💧 {h:.0f}%{Style.RESET_ALL}")

            # 4. Rain
            # Color rain: cyan if any rain
            rain_color = Fore.CYAN if r > 0 else Fore.RESET
            if r > 0:
                w_parts.append(f"{rain_color}🌧️ {r:.1f}mm{Style.RESET_ALL}")
            else:
                w_parts.append(f"{Style.DIM}Dry{Style.RESET_ALL}")
            
            # 5. Wind
            if w is not None and not math.isnan(w):
                # Color wind: yellow if strong (>20km/h)
                wind_color = Fore.YELLOW if w > 20 else Fore.RESET
                w_parts.append(f"{wind_color}💨 {w:.0f}km/h{Style.RESET_ALL}")
            
            # 6. Wet Indicator (Official Session Status)
            if is_wet:
                w_parts.append(f"{Fore.CYAN}{Style.BRIGHT}☔ [WET]{Style.RESET_ALL}")
            
            print(f"Weather: {'  '.join(w_parts)}")
        else:
            # Weather data has NaN values - show when it will be available
            if event_date:
                now = datetime.now(timezone.utc)
                days_until = (event_date - now).days
                forecast_available_in = max(0, days_until - 16)  # Open-Meteo forecasts ~16 days ahead
                if forecast_available_in > 0:
                    print(f"{Style.DIM}Weather: Unknown (forecast available in ~{forecast_available_in} days){Style.RESET_ALL}")
                else:
                    print(f"{Style.DIM}Weather: Unknown{Style.RESET_ALL}")
            else:
                print(f"{Style.DIM}Weather: Unknown{Style.RESET_ALL}")
    else:
        # No weather data at all (empty dict or None) - race is beyond forecast horizon
        if event_date:
            now = datetime.now(timezone.utc)
            days_until = (event_date - now).days
            forecast_available_in = max(0, days_until - 16)  # Open-Meteo forecasts ~16 days ahead
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
    is_race = sess in ("race", "sprint")
    win_label = "Pole" if is_quali else "Win"
    
    # Check if grid data is available for race/sprint sessions
    # (Checking generic is_race is enough if we generalized is_race above, but let's be explicit)
    has_grid = is_race and "grid" in df.columns and df["grid"].notna().any()
    
    # Print column headers
    if has_grid:
        header = (
            f"{Style.DIM}{'#':>3}   "
            f"{'Code':<4}  "
            f"{'Driver':<{max_name}}   "
            f"{'Team':<{max_team+2}}   "
            f"{'Grid':>4}   "
            f"{'Δ':>4}   "
            f"{'Avg':>5}   "
            f"{'Top3':>14}   "
            f"{win_label:>14}   "
            f"{'DNF':>14}   "
            f"{'Actual':>6}{Style.RESET_ALL}"
        )
    else:
        header = (
            f"{Style.DIM}{'#':>3}   "
            f"{'Code':<4}  "
            f"{'Driver':<{max_name}}   "
            f"{'Team':<{max_team+2}}   "
            f"{'Avg':>5}   "
            f"{'Top3':>14}   "
            f"{win_label:>14}   "
            f"{'DNF':>14}   "
            f"{'Actual':>6}{Style.RESET_ALL}"
        )
    print(header)
    
    # Horizontal separator
    # 3(#) + 3(sp) + 4(Code) + 2(sp) + max_name + 3(sp) + max_team+2 + 3(sp)
    # + [4(Grid) + 3(sp) + 4(Delta) + 3(sp)]
    # + 5(Avg) + 3(sp) + 14(Top3) + 3(sp) + 14(Win) + 3(sp) + 14(DNF) + 3(sp) + 6(Actual)
    base_width = 3 + 3 + 4 + 2 + max_name + 3 + max_team + 2 + 3 + 5 + 3 + 14 + 3 + 14 + 3 + 14 + 3 + 6
    if has_grid:
        sep_width = base_width + 4 + 3 + 4 + 3
    else:
        sep_width = base_width
    print(f"{Style.DIM}{'─' * sep_width}{Style.RESET_ALL}")
    
    for _, r in df.iterrows():
        pos = int(r["predicted_position"])
        name = sanitize_for_console(r.get("name") or "")[:max_name]
        code = sanitize_for_console(r.get("code") or "")[:3].upper()
        if not code: code = "???"

        team = sanitize_for_console(r.get("constructorName") or "")[:max_team]
        mp = float(r["mean_pos"])
        top3 = float(r["p_top3"]) * 100
        win = float(r["p_win"]) * 100
        dnf = float(r["p_dnf"]) * 100
        
        # Color coding for probabilities using helper
        win_color = _get_prob_color(win, is_dnf=False)
        top3_color = _get_prob_color(top3, is_dnf=False)
        dnf_color = _get_prob_color(dnf, is_dnf=True)

        # Visual bar for probabilities
        win_filled, win_empty = _render_bar_parts(win, width=5)
        top3_filled, top3_empty = _render_bar_parts(top3, width=5)
        dnf_filled, dnf_empty = _render_bar_parts(dnf, width=5)

        # Construct bars with colored filled part and dim empty part
        win_bar = f"{Style.DIM}▕{Style.RESET_ALL}{win_color}{win_filled}{Style.RESET_ALL}{Style.DIM}{win_empty}▏{Style.RESET_ALL}"
        top3_bar = f"{Style.DIM}▕{Style.RESET_ALL}{top3_color}{top3_filled}{Style.RESET_ALL}{Style.DIM}{top3_empty}▏{Style.RESET_ALL}"
        dnf_bar = f"{Style.DIM}▕{Style.RESET_ALL}{dnf_color}{dnf_filled}{Style.RESET_ALL}{Style.DIM}{dnf_empty}▏{Style.RESET_ALL}"
        
        # Grid position and delta for race sessions
        grid_str = ""
        delta_str = ""
        if has_grid:
            grid_pos = r.get("grid")
            if pd.notna(grid_pos):
                grid_int = int(grid_pos)
                delta = grid_int - pos  # positive = gained positions (started lower, finished higher)
                grid_str = f"{grid_int:>4}"
                if delta > 0:
                    delta_str = f"{Fore.GREEN}↑{delta:>2}{Style.RESET_ALL} "
                elif delta < 0:
                    delta_str = f"{Fore.RED}↓{abs(delta):>2}{Style.RESET_ALL} "
                else:
                    delta_str = f"{Style.DIM}  ={Style.RESET_ALL} "
            else:
                grid_str = f"{Style.DIM}  --{Style.RESET_ALL}"
                delta_str = f"{Style.DIM}  --{Style.RESET_ALL}"
        
        # Actual classification display
        actual_pos = r.get("actual_position")
        if pd.notna(actual_pos):
            classified_str = _render_actual_pos(pos, int(actual_pos))
        else:
            classified_str = f"{Style.DIM}{'--':>6}{Style.RESET_ALL}"
        
        # Print row
        row_str = (
            f"{Fore.YELLOW}{pos:>3}.{Style.RESET_ALL}  "
            f"{Style.BRIGHT}{code:<4}{Style.RESET_ALL}  "
            f"{Fore.CYAN}{name:<{max_name}}{Style.RESET_ALL}   "
            f"{Style.DIM}[{team:<{max_team}}]{Style.RESET_ALL}   "
        )

        if has_grid:
            row_str += f"{grid_str}   {delta_str}   "

        row_str += (
            f"{mp:5.1f}   "
            f"{top3_color}{top3:5.1f}% {top3_bar}{Style.RESET_ALL}   "
            f"{win_color}{win:5.1f}% {win_bar}{Style.RESET_ALL}   "
            f"{dnf_color}{dnf:5.1f}% {dnf_bar}{Style.RESET_ALL}   "
            f"{classified_str}"
        )
        print(row_str)

    # Print legend explaining abbreviations
    print(f"\n{Style.DIM}Legend: Avg=Predicted Mean Pos, Top3=Podium Prob, {win_label}=Win/Pole Prob, DNF=Retirement Prob, Actual=Result (✓=Exact, ≈=Close), Δ=Grid Delta{Style.RESET_ALL}")
