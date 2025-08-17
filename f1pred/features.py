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
from .roster import derive_roster

logger = get_logger(__name__)


def exponential_weights(dates: List[datetime], ref_date: datetime, half_life_days: int) -> np.ndarray:
    ages = np.array([(ref_date - d).days if isinstance(d, datetime) else 0 for d in dates], dtype=float)
    return np.power(0.5, ages / max(1, half_life_days))


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "driverId", "constructorId", "form_index", "team_form_index",
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


def collect_historical_results(jc: JolpicaClient, season: int, end_before: datetime, lookback_years: int = 50) -> pd.DataFrame:
    rows = []
    for yr in range(max(1950, season - lookback_years), season + 1):
        races = jc.get_season_schedule(str(yr))
        for r in races:
            date_str = r.get("date")
            time_str = r.get("time", "00:00:00Z")
            dt = datetime.fromisoformat((date_str or "1970-01-01") + "T" + time_str.replace("Z", "+00:00"))
            if dt >= end_before:
                continue
            season_s = r.get("season")
            rnd = r.get("round")
            circuit = r.get("Circuit", {}).get("circuitName")
            circuit_loc = r.get("Circuit", {}).get("Location", {})
            lat = circuit_loc.get("lat")
            lon = circuit_loc.get("long")
            results = jc.get_race_results(season_s, rnd)
            for res in results:
                drv = res.get("Driver", {})
                cons = res.get("Constructor", {})
                rows.append({
                    "season": int(season_s),
                    "round": int(rnd),
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
            qres = jc.get_qualifying_results(season_s, rnd)
            for res in qres:
                drv = res.get("Driver", {})
                cons = res.get("Constructor", {})
                pos = res.get("position")
                rows.append({
                    "season": int(season_s),
                    "round": int(rnd),
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
            sres = jc.get_sprint_results(season_s, rnd)
            for res in sres:
                drv = res.get("Driver", {})
                cons = res.get("Constructor", {})
                pos = res.get("position")
                rows.append({
                    "season": int(season_s),
                    "round": int(rnd),
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
    return pd.DataFrame(rows)


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
    return pd.DataFrame(rows)


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
    agg = dfg.groupby("driverId").apply(lambda g: pd.Series({
        "form_index": float((g["pos_score"] * g["w"]).sum() + (g["pts_score"] * g["w"]).sum()) / max(1e-6, g["w"].sum())
    })).reset_index()
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
        wagg = om.aggregate_for_session(wdf, start_dt, end_dt)
    except Exception as e:
        logger.info(f"[features] Weather fetch/aggregate failed: {e}")
        wagg = {}

    # Roster (single source of truth) — errors swallowed
    try:
        roster = build_roster(jc, str(season), str(rnd), event_dt=start_dt)
    except Exception as e:
        logger.info(f"[features] Roster derivation failed: {e}")
        roster = pd.DataFrame(columns=["driverId", "constructorId", "name"])

    # Historical results (errors swallowed)
    try:
        logger.info("[features] Collecting historical results")
        hist = collect_historical_results(jc, season=season, end_before=ref_date + timedelta(seconds=1), lookback_years=75)
    except Exception as e:
        logger.info(f"[features] Historical results fetch failed: {e}")
        hist = pd.DataFrame(columns=["driverId", "position", "date", "session", "constructorId", "points"])

    # Form indices
    form = compute_form_indices(hist, ref_date=ref_date, half_life_days=cfg.modelling.recency_half_life_days.base)

    # Team form indices
    team_form = hist[hist["session"] == "race"].dropna(subset=["constructorId", "date", "points"]) if not hist.empty else pd.DataFrame()
    if not team_form.empty:
        team_form["w"] = exponential_weights(list(team_form["date"]), ref_date, cfg.modelling.recency_half_life_days.team)
        team_idx = team_form.groupby("constructorId").apply(lambda g: pd.Series({
            "team_form_index": float((g["points"] * g["w"]).sum()) / max(1e-6, g["w"].sum())
        })).reset_index()
    else:
        team_idx = pd.DataFrame(columns=["constructorId", "team_form_index"])

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
        X = X.merge(weather_df, on="driverId", how="left")
        X["form_index"] = X["form_index"].fillna(0.0)
        X["team_form_index"] = X["team_form_index"].fillna(0.0)
        for col in ["wet_skill", "cold_skill", "wind_skill", "pressure_skill"]:
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