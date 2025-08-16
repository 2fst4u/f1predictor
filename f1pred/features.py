from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import numpy as np
import pandas as pd

from .util import get_logger
from .data.jolpica import JolpicaClient
from .data.open_meteo import OpenMeteoClient
from .data.openf1 import OpenF1Client
from .data.fastf1_backend import get_event, get_session_times

logger = get_logger(__name__)

def exponential_weights(dates: List[datetime], ref_date: datetime, half_life_days: int) -> np.ndarray:
    ages = np.array([(ref_date - d).days if isinstance(d, datetime) else 0 for d in dates], dtype=float)
    return np.power(0.5, ages / max(1, half_life_days))

def build_roster(jc: JolpicaClient, season: str, rnd: str) -> pd.DataFrame:
    try:
        q_res = jc.get_qualifying_results(season, rnd)
        if q_res:
            entries = []
            for r in q_res:
                d = r.get("Driver", {})
                c = r.get("Constructor", {})
                entries.append({
                    "driverId": d.get("driverId"),
                    "code": d.get("code"),
                    "name": f"{d.get('givenName')} {d.get('familyName')}",
                    "constructorId": c.get("constructorId"),
                    "constructorName": c.get("name"),
                    "number": d.get("permanentNumber"),
                })
            return pd.DataFrame(entries).drop_duplicates("driverId")
    except Exception:
        pass
    inferred = jc.derive_entry_list(season, rnd)
    df = pd.DataFrame(inferred).rename(columns={"givenName": "givenName", "familyName": "familyName"})
    if not df.empty:
        df["name"] = df.apply(lambda x: f"{x.get('givenName','')} {x.get('familyName','')}".strip(), axis=1)
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
    df = pd.DataFrame(rows)
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
    agg = dfg.groupby("driverId").apply(lambda g: pd.Series({
        "form_index": float((g["pos_score"] * g["w"]).sum() + (g["pts_score"] * g["w"]).sum()) / max(1e-6, g["w"].sum())
    })).reset_index()
    return agg

def build_session_features(jc: JolpicaClient, om: OpenMeteoClient, of1: Optional[OpenF1Client],
                           season: int, rnd: int, session_type: str,
                           ref_date: datetime,
                           cfg) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    schedule = jc.get_season_schedule(str(season))
    sched_row = [r for r in schedule if int(r.get("round")) == int(rnd)]
    if not sched_row:
        raise ValueError(f"No schedule for season={season} round={rnd}")
    race_info = sched_row[0]
    cir = race_info.get("Circuit", {})
    loc = cir.get("Location", {})
    lat, lon = float(loc.get("lat", 0.0)), float(loc.get("long", 0.0))
    circuit_name = cir.get("circuitName")
    start_dt = ref_date
    end_dt = ref_date + timedelta(hours=2)
    try:
        ev = get_event(season, rnd)
        if ev is not None:
            mapping = {
                "race": "Race",
                "qualifying": "Qualifying",
                "sprint": "Sprint",
                "sprint_qualifying": "Sprint Shootout"
            }
            st = get_session_times(ev, mapping.get(session_type, "Race"))
            if st:
                start_dt, end_dt = st
    except Exception:
        pass

    # Weather aggregation
    # Choose archive vs forecast based on ref_date vs now (use timezone-aware UTC)
    if ref_date.tzinfo is None:
        ref_date = ref_date.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    if ref_date < now:
        wdf = om.get_historical_weather(lat, lon, ref_date - timedelta(days=1), ref_date + timedelta(days=1))
        if wdf.empty:
            wdf = om.get_historical_forecast(lat, lon, ref_date - timedelta(days=1), ref_date + timedelta(days=1))
    else:
        wdf = om.get_forecast(lat, lon, ref_date - timedelta(days=1), ref_date + timedelta(days=1))
    wagg = om.aggregate_for_session(wdf, start_dt, end_dt)

    roster = build_roster(jc, str(season), str(rnd))
    hist = collect_historical_results(jc, season=season, end_before=ref_date + timedelta(seconds=1), lookback_years=75)
    form = compute_form_indices(hist, ref_date=ref_date, half_life_days=cfg.modelling.recency_half_life_days.base)

    team_form = hist[hist["session"] == "race"].dropna(subset=["constructorId", "date", "points"])
    if not team_form.empty:
        team_form["w"] = exponential_weights(list(team_form["date"]), ref_date, cfg.modelling.recency_half_life_days.team)
        team_idx = team_form.groupby("constructorId").apply(lambda g: pd.Series({
            "team_form_index": float((g["points"] * g["w"]).sum()) / max(1e-6, g["w"].sum())
        })).reset_index()
    else:
        team_idx = pd.DataFrame(columns=["constructorId", "team_form_index"])

    weather_idx = []
    for d in roster["driverId"].tolist():
        df_d = hist[(hist["driverId"] == d) & (hist["session"].isin(["race", "qualifying"]))].copy()
        if df_d.empty:
            weather_idx.append({"driverId": d, "wet_skill": 0.0, "cold_skill": 0.0, "wind_skill": 0.0, "pressure_skill": 0.0})
            continue
        df_d["pos"] = df_d.get("position", df_d.get("qpos")).astype("float")
        weather_idx.append({"driverId": d, "wet_skill": 0.0, "cold_skill": 0.0, "wind_skill": 0.0, "pressure_skill": 0.0})

    weather_df = pd.DataFrame(weather_idx)

    X = roster.merge(form, on="driverId", how="left").merge(team_idx, on="constructorId", how="left").merge(weather_df, on="driverId", how="left")
    X["form_index"] = X["form_index"].fillna(0.0)
    X["team_form_index"] = X["team_form_index"].fillna(0.0)
    for col in ["wet_skill", "cold_skill", "wind_skill", "pressure_skill"]:
        X[col] = X[col].fillna(0.0)

    X["session_type"] = session_type
    X["is_race"] = 1 if session_type == "race" else 0
    X["is_qualifying"] = 1 if session_type == "qualifying" or session_type == "sprint_qualifying" else 0
    X["is_sprint"] = 1 if "sprint" in session_type else 0

    for k, v in (wagg or {}).items():
        X[f"weather_{k}"] = v

    hh = teammate_head_to_head(hist)
    hh_rating = defaultdict(float)
    for _, row in hh.iterrows():
        a = row["driverA"]; b = row["driverB"]
        if row["a_better"] == 1:
            hh_rating[a] += 1.0
            hh_rating[b] -= 1.0
        else:
            hh_rating[a] -= 1.0
            hh_rating[b] += 1.0
    X["hh_index"] = X["driverId"].map(lambda d: hh_rating.get(d, 0.0))

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