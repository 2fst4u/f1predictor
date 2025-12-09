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
from .models import train_pace_model, estimate_dnf_probabilities
from .simulate import simulate_grid
from .report import generate_html_report
from .ensemble import EloModel, BradleyTerryModel, MixedEffectsLikeModel, EnsembleConfig, combine_pace

logger = get_logger(__name__)


# ... existing functions resolve_event, _session_title, _parse_lap_seconds, _get_actual_positions_for_session, _filter_sessions_for_round ...


def run_predictions_for_event(
    cfg,
    season: Optional[str],
    rnd: str,
    sessions: List[str],
    generate_html: bool = True,
    open_browser: bool = False,
    return_results: bool = False,
):
    """
    Generate predictions for given event, write CSV and optional HTML.

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
    sessions_out: List[Dict[str, Any]] = []

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
                pace_scale = float(getattr(getattr(cfg, "modelling", object()), "pace_scale", 0.6))
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

            # Ensemble skill components (all data-driven)
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
                ens_cfg = getattr(cfg.modelling, "ensemble", None)
                if ens_cfg is None:
                    ens_cfg = EnsembleConfig()
                else:
                    ens_cfg = EnsembleConfig(**ens_cfg.__dict__)
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

            prob_matrix_sorted = prob_matrix[order] if prob_matrix.size else prob_matrix
            pairwise_sorted = pairwise[np.ix_(order, order)] if pairwise.size else pairwise
            dnf_prob_sorted = dnf_prob[order]

            sess_rows: List[Dict[str, Any]] = []
            for _, row in ranked.iterrows():
                sess_rows.append(
                    {
                        "rank": int(row["predicted_position"]),
                        "name": row.get("name"),
                        "code": row.get("code") or "",
                        "team": row.get("constructorName") or "",
                        "mean_pos": float(row["mean_pos"]),
                        "p_top3": float(row["p_top3"]),
                        "p_win": float(row["p_win"]),
                        "p_dnf": float(row["p_dnf"]),
                        "actual_pos": int(row["actual_position"]) if pd.notna(row["actual_position"]) else None,
                        "delta": (int(row["delta"]) if pd.notna(row["delta"]) else None),
                    }
                )

            sessions_out.append(
                {
                    "session": sess,
                    "session_title": _session_title(sess),
                    "rows": sess_rows,
                }
            )

            session_results[sess] = {
                "ranked": ranked.copy(),
                "prob_matrix": prob_matrix_sorted.copy(),
                "pairwise": pairwise_sorted.copy(),
                "dnf_prob": dnf_prob_sorted.copy(),
                "meta": meta,
            }

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
                sessions_data=sessions_out,
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


# print_session_console unchanged
