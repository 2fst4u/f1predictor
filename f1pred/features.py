
"""Feature engineering for F1 predictions.

This module builds features from historical race data, weather, and driver/team
performance metrics. Features include form indices, weather sensitivity, and
teammate comparisons.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import pandas as pd

# Disable pandas warning about silent downcasting on fillna/ffill/bfill
pd.set_option("future.no_silent_downcasting", True)

from .util import get_logger
from .data.jolpica import JolpicaClient
from .data.open_meteo import OpenMeteoClient
from .data.openf1 import OpenF1Client
from .data.fastf1_backend import get_event, get_session_times
from .roster import derive_roster  # single source of truth for roster

__all__ = [
    "build_session_features",
    "collect_historical_results",
    "compute_form_indices",
    "compute_teammate_delta",
    "compute_grid_finish_delta",
    "exponential_weights",
]

logger = get_logger(__name__)

# In-process cache to avoid re-building history multiple times in the same run
# Key: (season, cutoff-date string, sorted roster driver ids)
_HIST_CACHE: Dict[Tuple[int, str, Tuple[str, ...]], pd.DataFrame] = {}

# Cache for historical weather per (season, round) to avoid repeated Open-Meteo calls in the same run
_WEATHER_EVENT_CACHE: Dict[Tuple[int, int], Dict[str, float]] = {}


def _cache_path_for_season(cache_dir: str, season: int) -> Path:
    """Get the parquet cache file path for a season."""
    return Path(cache_dir) / "history" / f"season_{season}.parquet"


def _load_season_cache(cache_dir: str, season: int) -> Optional[pd.DataFrame]:
    """Load cached season data from disk if it exists."""
    path = _cache_path_for_season(cache_dir, season)
    if path.exists():
        try:
            df = pd.read_parquet(path)
            logger.info(f"[features] [cache] Loaded {len(df)} rows from {path.name}")
            return df
        except Exception as e:
            logger.info(f"[features] [cache] Failed to load {path}: {e}")
    return None


def _save_season_cache(cache_dir: str, season: int, df: pd.DataFrame) -> None:
    """Save season data to disk cache."""
    if df.empty:
        return
    path = _cache_path_for_season(cache_dir, season)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        logger.info(f"[features] [cache] Saved {len(df)} rows to {path.name}")
    except Exception as e:
        logger.info(f"[features] [cache] Failed to save {path}: {e}")


def _weather_cache_path(cache_dir: str, season: int, rnd: int) -> Path:
    """Get the JSON cache file path for event weather."""
    return Path(cache_dir) / "weather" / f"event_{season}_{rnd}.json"


def _load_weather_cache(cache_dir: str, season: int, rnd: int) -> Optional[Dict[str, float]]:
    """Load cached weather aggregation from disk if it exists."""
    import json
    path = _weather_cache_path(cache_dir, season, rnd)
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
            logger.info(f"[features] [cache] Loaded weather from {path.name}")
            return data
        except Exception as e:
            logger.info(f"[features] [cache] Failed to load weather {path}: {e}")
    return None


def _save_weather_cache(cache_dir: str, season: int, rnd: int, data: Dict[str, float]) -> None:
    """Save weather aggregation to disk cache."""
    import json
    if not data:
        return
    path = _weather_cache_path(cache_dir, season, rnd)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info(f"[features] [cache] Saved weather to {path.name}")
    except Exception as e:
        logger.info(f"[features] [cache] Failed to save weather {path}: {e}")


def exponential_weights(dates: Union[List[datetime], pd.Series], ref_date: datetime, half_life_days: int) -> np.ndarray:
    if isinstance(dates, pd.Series):
        # Vectorized path for pandas Series (approx 4x faster)
        # Note: dates should not contain NaT ideally, but fillna(0) handles it safely
        ages = (ref_date - dates).dt.days.fillna(0).values.astype(float)
    else:
        # Legacy path for lists
        ages = np.array([(ref_date - d).days if isinstance(d, datetime) else 0 for d in dates], dtype=float)
    return np.power(0.5, ages / max(1, half_life_days))


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "driverId", "constructorId", "form_index", "team_form_index",
        "driver_team_form_index", "team_tenure_events",
        "weather_beta_temp", "weather_beta_pressure", "weather_beta_wind", "weather_beta_rain",
        "weather_effect",
        "temp_skill", "rain_skill", "wind_skill", "pressure_skill",
        "teammate_delta", "grid_finish_delta",
        "session_type", "is_race", "is_qualifying", "is_sprint"
    ])


def build_roster(jc: JolpicaClient, season: str, rnd: str, event_dt: Optional[datetime], of1: Optional[OpenF1Client] = None) -> pd.DataFrame:
    logger.info(f"[features] Deriving roster for {season} R{rnd}")
    t0 = time.time()
    entries = derive_roster(jc, season, rnd, event_dt=event_dt, now_dt=datetime.now(timezone.utc), openf1_client=of1)
    df = pd.DataFrame(entries)

    # Standardize essential columns so downstream code never fails
    if not df.empty:
        # human-readable name
        df["name"] = df.apply(lambda x: f"{(x.get('givenName') or '')} {(x.get('familyName') or '')}".strip(), axis=1)
        # permanentNumber -> number (string to preserve leading zeros if any)
        if "number" not in df.columns:
            df["number"] = df.get("permanentNumber")
        # safe defaults for code and constructorName
        if "code" not in df.columns:
            df["code"] = ""
        if "constructorName" not in df.columns:
            df["constructorName"] = ""
        # ensure driverId and constructorId exist as keys (fill if missing)
        if "driverId" not in df.columns:
            df["driverId"] = None
        if "constructorId" not in df.columns:
            df["constructorId"] = None

    logger.info(f"[features] Roster derived: {len(df)} drivers in {time.time() - t0:.2f}s")
    return df


def _parse_races_block(races: List[Dict[str, Any]], session_label: str, cutoff: datetime) -> pd.DataFrame:
    """Parse a block of races into a DataFrame, using vectorized date parsing for speed."""
    rows: List[Dict[str, Any]] = []

    # First pass: Extract raw fields (fast Python dict access)
    for r in races or []:
        season = int(r.get("season", 0))
        rnd = int(r.get("round", 0))
        date_str = r.get("date")
        time_str = r.get("time", "00:00:00Z")

        # Defer date parsing to vectorized step

        circuit = r.get("Circuit", {}).get("circuitName")
        loc = r.get("Circuit", {}).get("Location", {})
        lat = loc.get("lat")
        lon = loc.get("long")
        lat_val = float(lat) if lat else None
        lon_val = float(lon) if lon else None

        if session_label == "race":
            results = r.get("Results", []) or []
            for res in results:
                drv = res.get("Driver", {}) or {}
                cons = res.get("Constructor", {}) or {}
                rows.append({
                    "season": season,
                    "round": rnd,
                    "session": "race",
                    "date_str": date_str,
                    "time_str": time_str,
                    "circuit": circuit,
                    "lat": lat_val,
                    "lon": lon_val,
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
                    "season": season,
                    "round": rnd,
                    "session": "qualifying",
                    "date_str": date_str,
                    "time_str": time_str,
                    "circuit": circuit,
                    "lat": lat_val,
                    "lon": lon_val,
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
                    "season": season,
                    "round": rnd,
                    "session": "sprint",
                    "date_str": date_str,
                    "time_str": time_str,
                    "circuit": circuit,
                    "lat": lat_val,
                    "lon": lon_val,
                    "driverId": drv.get("driverId"),
                    "driverCode": drv.get("code"),
                    "constructorId": cons.get("constructorId"),
                    "position": int(pos) if pos else None,
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Vectorized date parsing (~5-10x faster than loop)
    # Fill missing dates/times for robust parsing
    df["date_str"] = df["date_str"].fillna("1970-01-01")
    df["time_str"] = df["time_str"].fillna("00:00:00Z")

    full_dates = df["date_str"] + "T" + df["time_str"].str.replace("Z", "+00:00", regex=False)

    # Use utc=True to enforce timezone awareness
    df["date"] = pd.to_datetime(full_dates, errors='coerce', utc=True)
    df["date"] = df["date"].fillna(datetime(1970, 1, 1, tzinfo=timezone.utc))

    # Filter
    df = df[df["date"] < cutoff]

    # Clean up temp columns
    df = df.drop(columns=["date_str", "time_str"])

    return df


def collect_historical_results(
    jc: JolpicaClient,
    season: int,
    end_before: datetime,
    lookback_years: int = 50,
    roster_driver_ids: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Optimized, roster-aware, season-bulk history builder with disk cache."""
    cutoff_key = end_before.date().isoformat()
    roster_key = tuple(sorted(roster_driver_ids)) if roster_driver_ids else tuple()
    cache_key = (season, cutoff_key, roster_key)
    if cache_key in _HIST_CACHE:
        return _HIST_CACHE[cache_key].copy()

    # Accumulate DataFrames instead of dicts to avoid serialization overhead
    dfs: List[pd.DataFrame] = []

    start_year = max(1950, season - lookback_years)
    logger.info(f"[features] [history] Scanning years {start_year}-{season}, cutoff < {end_before.isoformat()}")

    roster_set = set(roster_driver_ids or [])
    total_rows = 0
    current_year = datetime.now(timezone.utc).year

    for yr in range(season, start_year - 1, -1):
        # For completed seasons, try disk cache first
        is_current_season = (yr >= current_year)
        
        if cache_dir and not is_current_season:
            cached_df = _load_season_cache(cache_dir, yr)
            if cached_df is not None:
                # Filter by cutoff date
                cached_df = cached_df[cached_df["date"] < end_before]
                
                # Check roster match (Vectorized)
                matched = False
                if roster_set:
                    # ~50x faster than looping over dicts
                    matched = cached_df["driverId"].isin(roster_set).any()
                
                dfs.append(cached_df)
                total_rows += len(cached_df)
                logger.info(f"[features] [history] {yr}: loaded {len(cached_df)} rows from cache")
                
                if roster_set and not matched and yr < season:
                    logger.info(f"[features] [history] Stopping at {yr}: no results for current roster in this season")
                    break
                continue

        # Fetch from API
        try:
            races_blk = jc.get_season_race_results(str(yr))
            qual_blk = jc.get_season_qualifying_results(str(yr))
            sprint_blk = jc.get_season_sprint_results(str(yr))
        except Exception as e:
            logger.info(f"[features] [history] {yr}: bulk fetch failed: {e}; skipping year")
            continue

        r_df = _parse_races_block(races_blk, "race", end_before)
        q_df = _parse_races_block(qual_blk, "qualifying", end_before)
        s_df = _parse_races_block(sprint_blk, "sprint", end_before)

        matched = False
        if roster_set:
            for df_chk in (r_df, q_df, s_df):
                if not df_chk.empty:
                    if df_chk["driverId"].isin(roster_set).any():
                        matched = True
                        break

        # Append non-empty dataframes
        for df_part in (r_df, q_df, s_df):
            if not df_part.empty:
                dfs.append(df_part)
                total_rows += len(df_part)

        logger.info(
            f"[features] [history] {yr}: race={len(r_df)} qual={len(q_df)} sprint={len(s_df)} rows_total={total_rows}"
        )

        # Save completed seasons to disk cache
        if cache_dir and not is_current_season:
            yr_dfs = [d for d in (r_df, q_df, s_df) if not d.empty]
            if yr_dfs:
                yr_full_df = pd.concat(yr_dfs, ignore_index=True)
                _save_season_cache(cache_dir, yr, yr_full_df)

        if roster_set and not matched and yr < season:
            logger.info(f"[features] [history] Stopping at {yr}: no results for current roster in this season")
            break

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["driverId", "position", "date", "session", "constructorId", "points"])

    _HIST_CACHE[cache_key] = df.copy()
    alias_key = (season, cutoff_key, tuple())
    if alias_key not in _HIST_CACHE:
        _HIST_CACHE[alias_key] = df.copy()
    return df


def compute_form_indices(df: pd.DataFrame, ref_date: datetime, half_life_days: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["driverId", "form_index"])
    dfg = df[df["session"] == "race"].copy()
    dfg = dfg.dropna(subset=["driverId", "position", "date"])
    dfg["points"] = dfg["points"].fillna(0.0)
    dfg["pos_score"] = -dfg["position"].astype(float)
    dfg["pts_score"] = dfg["points"].astype(float)
    w = exponential_weights(dfg["date"], ref_date, half_life_days)
    dfg["w"] = w
    dfg["weighted_val"] = (dfg["pos_score"] + dfg["pts_score"]) * dfg["w"]
    sums = dfg.groupby("driverId")[["weighted_val", "w"]].sum().reset_index()
    sums["form_index"] = sums["weighted_val"] / sums["w"].clip(lower=1e-6)
    return sums[["driverId", "form_index"]]


def compute_driver_team_form(
    df: pd.DataFrame,
    roster: pd.DataFrame,
    ref_date: datetime,
    half_life_days: int,
    window_days: int = 730,
) -> pd.DataFrame:
    """Driver-team specific form for current constructor."""
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
    w = exponential_weights(races["date"], ref_date, half_life_days)
    races["w"] = w

    races["weighted_val"] = (races["pos_score"] + races["pts_score"]) * races["w"]

    # We need both the weighted sum/sum of weights AND the count of rows
    agg = races.groupby("driverId").agg(
        weighted_val_sum=("weighted_val", "sum"),
        w_sum=("w", "sum"),
        team_tenure_events=("driverId", "count")
    ).reset_index()

    agg["driver_team_form_index"] = agg["weighted_val_sum"] / agg["w_sum"].clip(lower=1e-6)

    return agg[["driverId", "driver_team_form_index", "team_tenure_events"]]


def compute_teammate_delta(
    hist: pd.DataFrame,
    ref_date: datetime,
    half_life_days: int,
) -> pd.DataFrame:
    """Recency-weighted average qualifying advantage vs team-mate.

    For each qualifying session where a driver has at least one team-mate
    (same constructor, same season/round), we compute the team average qpos
    and define delta = team_avg - driver_qpos so that positive = better than
    team average.
    """
    if hist.empty:
        return pd.DataFrame(columns=["driverId", "teammate_delta"])

    q = hist[hist["session"] == "qualifying"].copy()
    if q.empty:
        return pd.DataFrame(columns=["driverId", "teammate_delta"])

    q = q.dropna(subset=["constructorId", "driverId", "qpos", "date"])
    if q.empty:
        return pd.DataFrame(columns=["driverId", "teammate_delta"])

    q["qpos"] = q["qpos"].astype(float)
    w = exponential_weights(q["date"], ref_date, half_life_days)
    q["w"] = w

    # Count drivers per team-race group
    q["team_count"] = q.groupby(["season", "round", "constructorId"])["driverId"].transform("count")

    # Filter for groups with at least 2 drivers
    valid_q = q[q["team_count"] >= 2].copy()

    if valid_q.empty:
        return pd.DataFrame(columns=["driverId", "teammate_delta"])

    # Calculate team average qpos for each group
    valid_q["team_avg"] = valid_q.groupby(["season", "round", "constructorId"])["qpos"].transform("mean")

    # Calculate delta
    valid_q["delta"] = valid_q["team_avg"] - valid_q["qpos"]

    # Weighted aggregation
    valid_q["weighted_delta"] = valid_q["delta"] * valid_q["w"]

    # Group by driver and sum weighted delta and weights
    agg = valid_q.groupby("driverId")[["weighted_delta", "w"]].sum().reset_index()

    # Calculate final metric
    agg["teammate_delta"] = agg["weighted_delta"] / agg["w"].replace(0, 1e-6)

    return agg[["driverId", "teammate_delta"]]


def compute_grid_finish_delta(
    hist: pd.DataFrame,
    ref_date: datetime,
    half_life_days: int,
) -> pd.DataFrame:
    """Recency-weighted average (grid - finish) in races.

    positive = tends to gain positions (start further back than finish).
    """
    if hist.empty:
        return pd.DataFrame(columns=["driverId", "grid_finish_delta"])

    races = hist[hist["session"] == "race"].copy()
    races = races.dropna(subset=["driverId", "grid", "position", "date"])
    if races.empty:
        return pd.DataFrame(columns=["driverId", "grid_finish_delta"])

    races["gain"] = races["grid"].astype(float) - races["position"].astype(float)
    w = exponential_weights(races["date"], ref_date, half_life_days)
    races["w"] = w

    races["weighted_gain"] = races["gain"] * races["w"]
    sums = races.groupby("driverId")[["weighted_gain", "w"]].sum().reset_index()
    sums["grid_finish_delta"] = sums["weighted_gain"] / sums["w"].clip(lower=1e-6)
    return sums[["driverId", "grid_finish_delta"]]


def _aggregate_weather(om: OpenMeteoClient, lat: float, lon: float, event_dt: datetime) -> Dict[str, float]:
    """Aggregate Open-Meteo hourly weather around event_dt +/- 1 day."""
    try:
        wdf = om.get_historical_weather(lat, lon, event_dt - timedelta(days=1), event_dt + timedelta(days=1))
        agg = om.aggregate_for_session(wdf, event_dt - timedelta(hours=1), event_dt + timedelta(hours=1))
        return agg or {}
    except Exception:
        return {}


def compute_weather_sensitivity(
    om: OpenMeteoClient,
    hist: pd.DataFrame,
    roster: pd.DataFrame,
    ref_date: datetime,
    recent_years: int = 5
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Learn per-driver weather sensitivities from recent historical events."""
    if hist.empty or roster.empty:
        return pd.DataFrame(columns=[
            "driverId", "weather_beta_temp", "weather_beta_pressure", "weather_beta_wind", "weather_beta_rain"
        ]), {}

    min_year = max(1950, ref_date.year - recent_years)
    hist_recent = hist[(hist["date"].dt.year >= min_year)].copy()
    evt_keys = hist_recent[["season", "round", "circuit", "lat", "lon", "date"]].drop_duplicates()

    evt_weather: Dict[Tuple[int, int], Dict[str, float]] = {}
    for _, row in evt_keys.iterrows():
        try:
            lat = float(row.get("lat")) if row.get("lat") is not None else None
            lon = float(row.get("lon")) if row.get("lon") is not None else None
            if lat is None or lon is None:
                continue
            season_evt = int(row["season"]); round_evt = int(row["round"])
            if (season_evt, round_evt) in evt_weather:
                continue
            agg = _aggregate_weather(om, lat, lon, row["date"])
            evt_weather[(season_evt, round_evt)] = agg or {}
        except Exception:
            continue

    races = hist_recent[hist_recent["session"] == "race"].copy()
    if races.empty:
        return pd.DataFrame(columns=[
            "driverId", "weather_beta_temp", "weather_beta_pressure", "weather_beta_wind", "weather_beta_rain"
        ]), {}

    races = races.dropna(subset=["driverId", "position", "season", "round", "date"])
    races["pos_inv"] = -races["position"].astype(float)
    races["pos_inv_z"] = races.groupby("season")["pos_inv"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-6)
    )

    # Vectorized weather mapping
    weather_records = []
    for (season, rnd), w in evt_weather.items():
        weather_records.append({
            "season": season,
            "round": rnd,
            "weather_temp": w.get("temp_mean"),
            "weather_pressure": w.get("pressure_mean"),
            "weather_wind": w.get("wind_mean"),
            "weather_rain": w.get("rain_sum"),
        })

    if weather_records:
        w_lookup = pd.DataFrame(weather_records)
        races = races.merge(w_lookup, on=["season", "round"], how="left")
    else:
        for col in ["weather_temp", "weather_pressure", "weather_wind", "weather_rain"]:
            races[col] = np.nan

    weather_cols = ["weather_temp", "weather_pressure", "weather_wind", "weather_rain"]
    races = races.dropna(subset=weather_cols + ["pos_inv_z"])
    if races.empty:
        return pd.DataFrame(columns=[
            "driverId", "weather_beta_temp", "weather_beta_pressure", "weather_beta_wind", "weather_beta_rain"
        ]), {}

    # Vectorized weather sensitivity calculation
    # Filter drivers with fewer than 5 races (will default to 0.0 later via merge/fillna or explicit init)
    counts = races.groupby("driverId")["pos_inv_z"].count()
    valid_drivers = counts[counts >= 5].index

    # Initialize result for ALL drivers (so even those with <5 races get 0.0 rows)
    all_drivers = races["driverId"].unique()
    beta_df = pd.DataFrame({
        "driverId": all_drivers,
        "weather_beta_temp": 0.0,
        "weather_beta_pressure": 0.0,
        "weather_beta_wind": 0.0,
        "weather_beta_rain": 0.0
    }).set_index("driverId")

    if len(valid_drivers) > 0:
        valid_races = races[races["driverId"].isin(valid_drivers)].copy()

        # Ensure float type for correlation
        target_cols = ["weather_temp", "weather_pressure", "weather_wind", "weather_rain"]
        cols_to_corr = ["pos_inv_z"] + target_cols
        for c in cols_to_corr:
            valid_races[c] = valid_races[c].astype(float)

        # Compute correlations per driver
        # Result index: (driverId, feature_name), Columns: features
        corrs = valid_races.groupby("driverId")[cols_to_corr].corr()

        # Extract row corresponding to 'pos_inv_z' for each driver
        # xs(key, level) slices the MultiIndex
        try:
            pos_corrs = corrs.xs("pos_inv_z", level=1)

            # Select relevant columns and rename
            renamed = pos_corrs[target_cols].rename(columns={
                "weather_temp": "weather_beta_temp",
                "weather_pressure": "weather_beta_pressure",
                "weather_wind": "weather_beta_wind",
                "weather_rain": "weather_beta_rain"
            })

            # Fill NaNs with 0.0 (handles case where std dev is 0, which corr() returns as NaN)
            renamed = renamed.fillna(0.0)

            # Update the main DataFrame
            beta_df.update(renamed)
        except KeyError:
            # Should not happen if data exists, but safe fallback
            pass

    beta_df = beta_df.reset_index()

    return beta_df, {}


def build_session_features(jc: JolpicaClient, om: OpenMeteoClient, of1: Optional[OpenF1Client],
                           season: int, rnd: int, session_type: str,
                           ref_date: datetime,
                           cfg, extra_history: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
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
    
    # For historical events, try disk cache first
    wagg = None
    is_historical = ref_date < now
    if is_historical:
        wagg = _load_weather_cache(cfg.paths.cache_dir, season, rnd)
    
    if wagg is None:
        try:
            logger.info(f"[features] Fetching weather ({'historical' if is_historical else 'forecast'})")
            if is_historical:
                wdf = om.get_historical_weather(lat, lon, ref_date - timedelta(days=1), ref_date + timedelta(days=1))
                if wdf.empty:
                    wdf = om.get_historical_forecast(lat, lon, ref_date - timedelta(days=1), ref_date + timedelta(days=1))
            else:
                wdf = om.get_forecast(lat, lon, ref_date - timedelta(days=1), ref_date + timedelta(days=1))
            logger.info(f"[features] Weather fetched in {0 if wdf is None else len(wdf)} rows")
            wagg = om.aggregate_for_session(wdf, start_dt, end_dt)
            
            # Save to cache for historical events
            if is_historical and wagg:
                _save_weather_cache(cfg.paths.cache_dir, season, rnd, wagg)
        except Exception as e:
            logger.info(f"[features] Weather fetch/aggregate failed: {e}")
            wagg = {}

    # Roster
    try:
        roster = build_roster(jc, str(season), str(rnd), event_dt=start_dt, of1=of1)
    except Exception as e:
        logger.info(f"[features] Roster derivation failed: {e}")
        roster = pd.DataFrame(columns=["driverId", "constructorId", "name"])

    # Fetch actual starting grid for this race (if available)
    # Grid comes from race results endpoint - it shows the actual grid after penalties
    grid_df = pd.DataFrame(columns=["driverId", "grid"])
    if session_type == "race" and not roster.empty:
        try:
            race_results = jc.get_race_results(str(season), str(rnd))
            if race_results:
                grid_rows = []
                for res in race_results:
                    drv = res.get("Driver", {}) or {}
                    driver_id = drv.get("driverId")
                    grid_pos = res.get("grid")
                    if driver_id and grid_pos is not None:
                        grid_rows.append({
                            "driverId": driver_id,
                            "grid": int(grid_pos) if grid_pos else None
                        })
                if grid_rows:
                    grid_df = pd.DataFrame(grid_rows)
                    logger.info(f"[features] Fetched actual grid for {len(grid_df)} drivers")
            else:
                logger.info("[features] Race results not yet available - grid will be NaN")
        except Exception as e:
            logger.info(f"[features] Could not fetch grid from race results: {e}")


    # Historical results (optimized, cached, roster-aware)
    logger.info("[features] Collecting historical results")
    t3 = time.time()
    try:
        roster_ids = roster["driverId"].dropna().astype(str).tolist() if not roster.empty else []
        hist = collect_historical_results(jc, season=season, end_before=ref_date + timedelta(seconds=1),
                                          lookback_years=75, roster_driver_ids=roster_ids,
                                          cache_dir=cfg.paths.cache_dir)
        
        # Inject extra intra-weekend history if provided
        if extra_history is not None and not extra_history.empty:
            logger.info(f"[features] Appending {len(extra_history)} rows of extra history")
            # Ensure proper concatenation
            hist = pd.concat([hist, extra_history], ignore_index=True)
            
        logger.info(f"[features] [history] Collection finished in {time.time() - t3:.2f}s (rows={len(hist)})")
    except Exception as e:
        logger.info(f"[features] Historical results fetch failed: {e}")
        hist = pd.DataFrame(columns=["driverId", "position", "date", "session", "constructorId", "points"])

    # Form indices
    form = compute_form_indices(hist, ref_date=ref_date, half_life_days=cfg.modelling.recency_half_life_days.base)

    # Ensure 'session' column exists
    if 'session' not in hist.columns:
        logger.warning("[features] 'session' column missing from historical data.")
        hist['session'] = pd.Series(dtype='str')  # Add empty 'session' column

    # Ensure necessary columns exist
    required_columns = ["constructorId", "date", "points"]
    missing_columns = [col for col in required_columns if col not in hist.columns]
    if missing_columns:
        logger.warning(f"[features] Missing columns from historical data: {missing_columns}")
        for col in missing_columns:
            hist[col] = pd.Series(dtype='str')  # Add missing columns with appropriate dtype

    team_form = hist[hist["session"] == "race"].dropna(subset=["constructorId", "date", "points"])
    if not team_form.empty:
        team_form["w"] = exponential_weights(team_form["date"], ref_date, cfg.modelling.recency_half_life_days.team)
        team_form["weighted_points"] = team_form["points"] * team_form["w"]

        sums = team_form.groupby("constructorId")[["weighted_points", "w"]].sum().reset_index()
        sums["team_form_index"] = sums["weighted_points"] / sums["w"].clip(lower=1e-6)
        team_idx = sums[["constructorId", "team_form_index"]]
    else:
        team_idx = pd.DataFrame(columns=["constructorId", "team_form_index"])

    drv_team_form = compute_driver_team_form(
        hist, roster, ref_date=ref_date, half_life_days=cfg.modelling.recency_half_life_days.team, window_days=730
    )

    # New dynamic features: teammate_delta & grid_finish_delta
    try:
        tm_delta = compute_teammate_delta(
            hist,
            ref_date=ref_date,
            half_life_days=cfg.modelling.recency_half_life_days.base,
        )
        logger.info("[features] teammate_delta computed for %d drivers", len(tm_delta))
    except Exception as e:
        logger.info(f"[features] teammate_delta computation failed: {e}")
        tm_delta = pd.DataFrame(columns=["driverId", "teammate_delta"])

    try:
        gf_delta = compute_grid_finish_delta(
            hist,
            ref_date=ref_date,
            half_life_days=cfg.modelling.recency_half_life_days.base,
        )
        logger.info("[features] grid_finish_delta computed for %d drivers", len(gf_delta))
    except Exception as e:
        logger.info(f"[features] grid_finish_delta computation failed: {e}")
        gf_delta = pd.DataFrame(columns=["driverId", "grid_finish_delta"])

    # Weather sensitivity per driver
    try:
        beta_df, _ = compute_weather_sensitivity(om, hist, roster, ref_date=ref_date, recent_years=5)
    except Exception:
        beta_df = pd.DataFrame(columns=[
            "driverId", "weather_beta_temp", "weather_beta_pressure", "weather_beta_wind", "weather_beta_rain"
        ])

    # Build weather skill scores from beta values
    # The beta values ARE the skills - they represent correlation between
    # driver performance and weather conditions (positive = better in that condition)
    if not beta_df.empty and not roster.empty:
        weather_df = roster[["driverId"]].merge(beta_df, on="driverId", how="left")
        weather_df["temp_skill"] = weather_df["weather_beta_temp"].fillna(0.0)
        weather_df["rain_skill"] = weather_df["weather_beta_rain"].fillna(0.0)
        weather_df["wind_skill"] = weather_df["weather_beta_wind"].fillna(0.0)
        weather_df["pressure_skill"] = weather_df["weather_beta_pressure"].fillna(0.0)
        weather_df = weather_df[["driverId", "temp_skill", "rain_skill", "wind_skill", "pressure_skill"]]
    else:
        weather_df = pd.DataFrame({
            "driverId": roster["driverId"] if not roster.empty else [],
            "temp_skill": 0.0, "rain_skill": 0.0, "wind_skill": 0.0, "pressure_skill": 0.0
        })

    # Merge features
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
        X = X.merge(beta_df, on="driverId", how="left")
        X = X.merge(weather_df, on="driverId", how="left")
        X = X.merge(tm_delta, on="driverId", how="left")
        X = X.merge(gf_delta, on="driverId", how="left")
        
        # Merge grid position (will be NaN if not yet available)
        if not grid_df.empty:
            X = X.merge(grid_df, on="driverId", how="left")
        else:
            X["grid"] = np.nan



        # Dynamically derive defaults from the distribution of knowns
        # For form index (score where higher = better, e.g. -1.0 > -20.0), a rookie should get a NEUTRAL score (median).
        # This allows their car performance (team features) to determine their initial predicted position.
        neutral_form = form["form_index"].median() if not form.empty else -10.0
        neutral_team_form = team_idx["team_form_index"].median() if not team_idx.empty else -10.0
        neutral_drv_team_form = drv_team_form["driver_team_form_index"].median() if not drv_team_form.empty else -10.0
        
        # For deltas (teammate comparison), default to the median observed delta 
        # to assume a rookie is roughly average relative to their teammate initially.
        neutral_teammate_delta = tm_delta["teammate_delta"].median() if not tm_delta.empty else 0.0
        # Grid finish delta: similarly, assume average racecraft
        neutral_gf_delta = gf_delta["grid_finish_delta"].median() if not gf_delta.empty else 0.0
        
        defaults = {
            "form_index": neutral_form,
            "team_form_index": neutral_team_form,
            "driver_team_form_index": neutral_drv_team_form,
            "team_tenure_events": 0.0,
            # Weather betas: 0.0 is fine (no correlation)
            "weather_beta_temp": 0.0, 
            "weather_beta_pressure": 0.0, 
            "weather_beta_wind": 0.0, 
            "weather_beta_rain": 0.0,
            "temp_skill": 0.0, 
            "rain_skill": 0.0, 
            "wind_skill": 0.0, 
            "pressure_skill": 0.0,
            # Deltas: assume neutral behavior
            "teammate_delta": neutral_teammate_delta,
            "grid_finish_delta": neutral_gf_delta,
        }
        
        for col, default_val in defaults.items():
            if col in X.columns:
                X[col] = X[col].fillna(default_val)
                
        wt = wagg or {}
        
        # Calculate weather effect safely (treating missing as 0.0)
        t_mean = float(wt.get("temp_mean", 0.0))
        p_mean = float(wt.get("pressure_mean", 0.0))
        w_mean = float(wt.get("wind_mean", 0.0))
        r_sum = float(wt.get("rain_sum", 0.0))
        
        if t_mean != t_mean: t_mean = 0.0
        if p_mean != p_mean: p_mean = 0.0
        if w_mean != w_mean: w_mean = 0.0
        if r_sum != r_sum: r_sum = 0.0

        X["weather_effect"] = (
            X.get("weather_beta_temp", 0.0) * t_mean +
            X.get("weather_beta_pressure", 0.0) * p_mean +
            X.get("weather_beta_wind", 0.0) * w_mean +
            X.get("weather_beta_rain", 0.0) * r_sum
        )

    X["session_type"] = session_type
    X["is_race"] = 1 if session_type == "race" else 0
    X["is_qualifying"] = 1 if session_type in ("qualifying", "sprint_qualifying") else 0
    X["is_sprint"] = 1 if "sprint" in session_type else 0

    # Only add weather columns if they are not NaN
    for k, v in (wagg or {}).items():
        if v == v and v is not None:  # Check for not-NaN
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
