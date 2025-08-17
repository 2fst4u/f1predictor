from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import time

import numpy as np
import pandas as pd

from .util import get_logger
from .data.jolpica import JolpicaClient
from .data.open_meteo import OpenMeteoClient
from .data.openf1 import OpenF1Client
from .data.fastf1_backend import get_event, get_session_times
from .roster import derive_roster  # single source of truth

logger = get_logger(__name__)

# Simple in-process cache to avoid re-building history in the same run
# Keyed by (season, cutoff-date string, sorted roster driver ids)
_HIST_CACHE: Dict[Tuple[int, str, Tuple[str, ...]], pd.DataFrame] = {}


def exponential_weights(dates: List[datetime], ref_date: datetime, half_life_days: int) -> np.ndarray:
    ages = np.array([(ref_date - d).days if isinstance(d, datetime) else 0 for d in dates], dtype=float)
    return np.power(0.5, ages / max(1, half_life_days))


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "driverId", "constructorId", "form_index", "team_form_index",
        "driver_team_form_index", "team_tenure_events",
        "wet_skill", "cold_skill", "wind_skill", "pressure_skill",
        "session_type", "is_race", "is_qualifying", "is_sprint"
    ])


def build_roster(jc: JolpicaClient, season: str, rnd: str, event_dt: Optional[datetime]) -> pd.DataFrame:
    logger.info(f"[features] Deriving roster for {season} R{rnd}")
    t0 = time.time()
    entries = derive_roster(jc, season, rnd, event_dt=event_dt, now_dt=datetime.now(timezone.utc))
    df = pd.DataFrame(entries)
    if not df.empty:
        df["name"] = df.apply(lambda x: f"{(x.get('givenName') or '')} {(x.get('familyName') or '')}".strip(), axis=1)
    logger.info(f"[features] Roster derived: {len(df)} drivers in {time.time() - t0:.2f}s")
    return df


def _parse_races_block(races: List[Dict[str, Any]], session_label: str, cutoff: datetime) -> List[Dict[str, Any]]:
    rows = []
    for r in races or []:
        date_str = r.get("date")
        time_str = r.get("time", "00:00:00Z")
        try:
            dt = datetime.fromisoformat((date_str or "1970-01-01") + "T" + time_str.replace("Z", "+00:00"))
        except Exception:
            dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
        if dt >= cutoff:
            continue
        circuit = r.get("Circuit", {}).get("circuitName")
        loc = r.get("Circuit", {}).get("Location", {})
        lat = loc.get("lat")
        lon = loc.get("long")

        if session_label == "race":
            results = r.get("Results", []) or []
            for res in results:
                drv = res.get("Driver", {}) or {}
                cons = res.get("Constructor", {}) or {}
                rows.append({
                    "season": int(r.get("season")),
                    "round": int(r.get("round")),
                    "session": "race",
                    "date": dt,
                    "circuit": circuit,
                    "lat": float(lat) if lat else None,
                    "lon": float(lon) if lon else None,
                    "driverId": drv.get("driverId"),
                    "driverCode": drv.get("code"),
                    "constructorId": cons.get("constructorId"),
                    "grid": int(res.get("grid", 0) or 0),
                    "position": int(res.get("position", 0) or 0) if res.get("position") else None,
                    "status": res.get("status"),
                    "points": float(res.get("points", 0.0) or 0.0),
                    "fastestLap": (res.get("FastestLap") or {}).get("rank"),
                })
        elif session_label == "qualifying":
            qres = r.get("QualifyingResults", []) or []
            for res in qres:
                drv = res.get("Driver", {}) or {}
                cons = res.get("Constructor", {}) or {}
                pos = res.get("position")
                rows.append({
                    "season": int(r.get("season")),
                    "round": int(r.get("round")),
                    "session": "qualifying",
                    "date": dt,
                    "circuit": circuit,
                    "lat": float(lat) if lat else None,
                    "lon": float(lon) if lon else None,
                    "driverId": drv.get("driverId"),
                    "driverCode": drv.get("code"),
                    "constructorId": cons.get("constructorId"),
                    "qpos": int(pos) if pos else None,
                    "q1": (res.get("Q1") or None),
                    "q2": (res.get("Q2") or None),
                    "q3": (res.get("Q3") or None),
                })
        elif session_label == "sprint":
            sres = r.get("SprintResults", []) or []
            for res in sres:
                drv = res.get("Driver", {}) or {}
                cons = res.get("Constructor", {}) or {}
                pos = res.get("position")
                rows.append({
                    "season": int(r.get("season")),
                    "round": int(r.get("round")),
                    "session": "sprint",
                    "date": dt,
                    "circuit": circuit,
                    "lat": float(lat) if lat else None,
                    "lon": float(lon) if lon else None,
                    "driverId": drv.get("driverId"),
                    "driverCode": drv.get("code"),
                    "constructorId": cons.get("constructorId"),
                    "position": int(pos) if pos else None,
                })
    return rows


def collect_historical_results(
    jc: JolpicaClient,
    season: int,
    end_before: datetime,
    lookback_years: int = 50,
    roster_driver_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Optimized, roster-aware, season-bulk history builder.
    - Uses bulk season endpoints (1–3 calls per season) to avoid 429s and reduce API pressure.
    - Iterates seasons backwards from 'season' down to max(1950, season - lookback_years).
    - Early stop: stop at the first season where none of the current roster drivers appear at all.
    - Always respects time-cut end_before.
    - In-process cached by (season, cutoff_date_str, roster_ids) to avoid duplicate work within the same run.
    """
    cutoff_key = end_before.date().isoformat()
    roster_key = tuple(sorted(roster_driver_ids)) if roster_driver_ids else tuple()
    cache_key = (season, cutoff_key, roster_key)
    if cache_key in _HIST_CACHE:
        return _HIST_CACHE[cache_key].copy()

    rows: List[Dict[str, Any]] = []
    start_year = max(1950, season - lookback_years)
    logger.info(f"[features] [history] Scanning years {start_year}-{season}, cutoff < {end_before.isoformat()}")

    roster_set = set(roster_driver_ids or [])
    total_rows = 0

    for yr in range(season, start_year - 1, -1):
        try:
            races_blk = jc.get_season_race_results(str(yr))
            qual_blk = jc.get_season_qualifying_results(str(yr))
            sprint_blk = jc.get_season_sprint_results(str(yr))
        except Exception as e:
            logger.info(f"[features] [history] {yr}: bulk fetch failed: {e}; skipping year")
            continue

        r_rows = _parse_races_block(races_blk, "race", end_before)
        q_rows = _parse_races_block(qual_blk, "qualifying", end_before)
        s_rows = _parse_races_block(sprint_blk, "sprint", end_before)

        # Check roster coverage for this season (before appending)
        matched = 0
        if roster_set:
            for block in (r_rows, q_rows, s_rows):
                for rr in block:
                    if rr.get("driverId") in roster_set:
                        matched = 1
                        break
                if matched:
                    break

        # Append parsed rows
        rows.extend(r_rows); total_rows += len(r_rows)
        rows.extend(q_rows); total_rows += len(q_rows)
        rows.extend(s_rows); total_rows += len(s_rows)

        logger.info(
            f"[features] [history] {yr}: race={len(r_rows)} qual={len(q_rows)} sprint={len(s_rows)} rows_total={total_rows}"
        )

        # Early stop: if no roster drivers appear at all in this season, stop walking further back
        if roster_set and matched == 0:
            logger.info(f"[features] [history] Stopping at {yr}: no results for current roster in this season")
            break

    df = pd.DataFrame(rows)
    _HIST_CACHE[cache_key] = df.copy()
    # Also store an alias under the general key (season, cutoff, empty roster) so a second call
    # without roster ids (e.g., from DNF step) hits the cache and avoids re-fetching.
    alias_key = (season, cutoff_key, tuple())
    if alias_key not in _HIST_CACHE:
        _HIST_CACHE[alias_key] = df.copy()
    return df


def teammate_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    q = df[df["session"] == "qualifying"].copy()
    if q.empty:
        return pd.DataFrame()
    q["qpos"] = q["qpos"].astype("float")
    q = q.dropna(subset=["constructorId", "driverId", "qpos", "date"])
    groups = q.groupby(["season", "round", "constructorId"])
    rows = []
    for (season, rnd, cons), g in groups:
        if g.shape[0] < 2:
            continue
        g = g.sort_values("qpos")
        drivers = list(g["driverId"])
        positions = list(g["qpos"])
        date = g["date"].max()
        for i in range(len(drivers)):
            for j in range(i + 1, len(drivers)):
                rows.append({"date": date, "driverA": drivers[i], "driverB": drivers[j],
                             "a_better": 1 if positions[i] < positions[j] else 0})
    hh = pd.DataFrame(rows)
    return hh


def compute_form_indices(df: pd.DataFrame, ref_date: datetime, half_life_days: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["driverId", "form_index"])
    dfg = df[df["session"] == "race"].copy()
    dfg = dfg.dropna(subset=["driverId", "position", "date"])
    dfg["points"] = dfg["points"].fillna(0.0)
    dfg["pos_score"] = -dfg["position"].astype(float)
    dfg["pts_score"] = dfg["points"].astype(float)
    w = exponential_weights(list(dfg["date"]), ref_date, half_life_days)
    dfg["w"] = w
    agg = dfg.groupby("driverId").apply(
        lambda g: pd.Series({
            "form_index": float((g["pos_score"] * g["w"]).sum() + (g["pts_score"] * g["w"]).sum()) / max(1e-6, g["w"].sum())
        }),
        include_groups=False
    ).reset_index()
    return agg


def compute_driver_team_form(
    df: pd.DataFrame,
    roster: pd.DataFrame,
    ref_date: datetime,
    half_life_days: int,
    window_days: int = 730,
) -> pd.DataFrame:
    """
    Driver-team specific form (recency-weighted, last 'window_days') using only events
    where the driver's event constructor equals their current roster constructor.

    Returns DataFrame: [driverId, driver_team_form_index, team_tenure_events]
    """
    if df.empty or roster.empty:
        return pd.DataFrame(columns=["driverId", "driver_team_form_index", "team_tenure_events"])

    min_dt = ref_date - timedelta(days=window_days)
    races = df[(df["session"] == "race") & (df["date"] >= min_dt)].copy()
    races = races.dropna(subset=["driverId", "constructorId", "date"])

    cur_map = roster.set_index("driverId")["constructorId"].to_dict()
    races["cur_constructor"] = races["driverId"].map(cur_map)
    races = races[races["constructorId"] == races["cur_constructor"]].copy()
    if races.empty:
        return pd.DataFrame(columns=["driverId", "driver_team_form_index", "team_tenure_events"])

    races["pos_score"] = -races["position"].astype(float).fillna(0.0)
    races["pts_score"] = races["points"].astype(float).fillna(0.0)
    w = exponential_weights(list(races["date"]), ref_date, half_life_days)
    races["w"] = w

    agg = races.groupby("driverId").apply(
        lambda g: pd.Series({
            "driver_team_form_index": float((g["pos_score"] * g["w"]).sum() + (g["pts_score"] * g["w"]).sum()) / max(1e-6, g["w"].sum()),
            "team_tenure_events": int(g.shape[0])
        }),
        include_groups=False
    ).reset_index()
    return agg


def build_session_features(jc: JolpicaClient, om: OpenMeteoClient, of1: Optional[OpenF1Client],
                           season: int, rnd: int, session_type: str,
                           ref_date: datetime,
                           cfg) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    logger.info(f"[features] Fetching schedule for {season} R{rnd}")
    t_all = time.time()

    # Schedule; if missing, return empty features (never raise)
    try:
        schedule = jc.get_season_schedule(str(season))
        sched_row = [r for r in schedule if int(r.get("round")) == int(rnd)]
        if not sched_row:
            logger.info(f"[features] No schedule row for {season} R{rnd}; returning empty features")
            meta = {
                "season": season, "round": rnd, "circuit": None,
                "lat": None, "lon": None, "session_start": ref_date, "session_end": ref_date + timedelta(hours=2),
                "weather": {}
            }
            return _empty_feature_frame(), meta, pd.DataFrame(columns=["driverId", "constructorId", "name"])
        race_info = sched_row[0]
        cir = race_info.get("Circuit", {})
        loc = cir.get("Location", {})
        lat, lon = float(loc.get("lat", 0.0)), float(loc.get("long", 0.0))
        circuit_name = cir.get("circuitName")
    except Exception as e:
        logger.info(f"[features] Schedule fetch failed: {e}; returning empty features")
        meta = {
            "season": season, "round": rnd, "circuit": None,
            "lat": None, "lon": None, "session_start": ref_date, "session_end": ref_date + timedelta(hours=2),
            "weather": {}
        }
        return _empty_feature_frame(), meta, pd.DataFrame(columns=["driverId", "constructorId", "name"])

    # Session timing (FastF1 attempt, then default)
    start_dt = ref_date
    end_dt = ref_date + timedelta(hours=2)
    try:
        logger.info("[features] Attempting FastF1 session timing")
        ev = get_event(season, rnd)
        st = get_session_times(ev, {"race": "Race", "qualifying": "Qualifying", "sprint": "Sprint",
                                    "sprint_qualifying": "Sprint Shootout"}.get(session_type, "Race")) if ev else None
        if st:
            start_dt, end_dt = st
            logger.info("[features] FastF1 timing resolved")
        else:
            logger.info("[features] FastF1 timing unavailable; using default window")
    except Exception:
        logger.info("[features] FastF1 timing failed; using default window")

    # Weather aggregation (errors swallowed; empty weather allowed)
    if ref_date.tzinfo is None:
        ref_date = ref_date.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    try:
        logger.info(f"[features] Fetching weather ({'historical' if ref_date < now else 'forecast'})")
        if ref_date < now:
            wdf = om.get_historical_weather(lat, lon, ref_date - timedelta(days=1), ref_date + timedelta(days=1))
            if wdf.empty:
                wdf = om.get_historical_forecast(lat, lon, ref_date - timedelta(days=1), ref_date + timedelta(days=1))
        else:
            wdf = om.get_forecast(lat, lon, ref_date - timedelta(days=1), ref_date + timedelta(days=1))
        logger.info(f"[features] Weather fetched in {0 if wdf is None else len(wdf)} rows")
        wagg = om.aggregate_for_session(wdf, start_dt, end_dt)
    except Exception as e:
        logger.info(f"[features] Weather fetch/aggregate failed: {e}")
        wagg = {}

    # Roster
    try:
        roster = build_roster(jc, str(season), str(rnd), event_dt=start_dt)
    except Exception as e:
        logger.info(f"[features] Roster derivation failed: {e}")
        roster = pd.DataFrame(columns=["driverId", "constructorId", "name"])

    # Historical results (optimized, cached, roster-aware)
    logger.info("[features] Collecting historical results")
    t3 = time.time()
    try:
        roster_ids = roster["driverId"].dropna().astype(str).tolist() if not roster.empty else []
        hist = collect_historical_results(jc, season=season, end_before=ref_date + timedelta(seconds=1),
                                          lookback_years=75, roster_driver_ids=roster_ids)
        logger.info(f"[features] [history] Collection finished in {time.time() - t3:.2f}s (rows={len(hist)})")
    except Exception as e:
        logger.info(f"[features] Historical results fetch failed: {e}")
        hist = pd.DataFrame(columns=["driverId", "position", "date", "session", "constructorId", "points"])

    # Generic driver form and team form indices
    form = compute_form_indices(hist, ref_date=ref_date, half_life_days=cfg.modelling.recency_half_life_days.base)

    team_form = hist[hist["session"] == "race"].dropna(subset=["constructorId", "date", "points"])
    if not team_form.empty:
        team_form["w"] = exponential_weights(list(team_form["date"]), ref_date, cfg.modelling.recency_half_life_days.team)
        team_idx = team_form.groupby("constructorId").apply(
            lambda g: pd.Series({
                "team_form_index": float((g["points"] * g["w"]).sum()) / max(1e-6, g["w"].sum())
            }),
            include_groups=False
        ).reset_index()
    else:
        team_idx = pd.DataFrame(columns=["constructorId", "team_form_index"])

    # Driver-team specific form (current constructor only)
    drv_team_form = compute_driver_team_form(
        hist, roster, ref_date=ref_date, half_life_days=cfg.modelling.recency_half_life_days.team, window_days=730
    )

    # Weather sensitivity placeholder features
    weather_df = pd.DataFrame({
        "driverId": roster["driverId"] if not roster.empty else [],
        "wet_skill": 0.0, "cold_skill": 0.0, "wind_skill": 0.0, "pressure_skill": 0.0
    })

    # Merge features (handle empty frames gracefully)
    X = _empty_feature_frame()
    if not roster.empty:
        X = roster.copy()
        if "driverId" not in X.columns:
            X["driverId"] = None
        if "constructorId" not in X.columns:
            X["constructorId"] = None
        X = X.merge(form, on="driverId", how="left")
        X = X.merge(team_idx, on="constructorId", how="left")
        X = X.merge(drv_team_form, on="driverId", how="left")
        X = X.merge(weather_df, on="driverId", how="left")
        for col in ["form_index", "team_form_index", "driver_team_form_index", "team_tenure_events",
                    "wet_skill", "cold_skill", "wind_skill", "pressure_skill"]:
            if col in X.columns:
                X[col] = X[col].fillna(0.0)

    # Session type encoding
    X["session_type"] = session_type
    X["is_race"] = 1 if session_type == "race" else 0
    X["is_qualifying"] = 1 if session_type in ("qualifying", "sprint_qualifying") else 0
    X["is_sprint"] = 1 if "sprint" in session_type else 0

    # Weather features
    for k, v in (wagg or {}).items():
        X[f"weather_{k}"] = v

    logger.info(f"[features] Feature build complete in {time.time() - t_all:.2f}s")

    meta = {
        "season": season,
        "round": rnd,
        "circuit": circuit_name,
        "lat": lat,
        "lon": lon,
        "session_start": start_dt,
        "session_end": end_dt,
        "weather": wagg,
    }
    return X, meta, roster