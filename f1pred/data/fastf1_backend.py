from __future__ import annotations
from typing import Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)
_CACHE_INITED = False

# Maximum seconds a single FastF1 session load may take before we give up.
# Prevents hangs when F1 live-timing servers are unresponsive.
_SESSION_LOAD_TIMEOUT = 15


def init_fastf1(cache_dir: str) -> bool:
    """Enable cache and silence FastF1 noisy logs; never raise."""
    global _CACHE_INITED
    try:
        import fastf1
    except ImportError:
        logger.warning("FastF1 is not installed; fastf1_backend is disabled.")
        return False

    if _CACHE_INITED:
        return True
    try:
        fastf1.Cache.enable_cache(cache_dir)
        # Silence FastF1 INFO/DEBUG from third-party
        try:
            from fastf1.logger import set_log_level as ff1_set_log_level  # type: ignore
            ff1_set_log_level("ERROR")
        except Exception:
            pass
        # Also clamp specific loggers (best-effort)
        for name in ("fastf1", "fastf1.api", "fastf1.core", "fastf1.fastf1.core"):
            try:
                logging.getLogger(name).setLevel(logging.ERROR)
            except Exception:
                pass
        _CACHE_INITED = True
        return True
    except Exception as e:
        logger.warning(f"FastF1 cache init failed: {e}")
        return False


def _to_py_datetime(dt_obj: Any) -> Optional[datetime]:
    try:
        if hasattr(dt_obj, "to_pydatetime"):
            py = dt_obj.to_pydatetime()
        else:
            py = dt_obj
        if not isinstance(py, datetime):
            return None
        if py.tzinfo is None:
            py = py.replace(tzinfo=timezone.utc)
        else:
            try:
                py = py.astimezone(timezone.utc)
            except Exception:
                pass
        return py
    except Exception:
        return None


def get_event(season: int, round_no: int):
    try:
        import fastf1
        return fastf1.get_event(season, round_no)
    except Exception:
        return None


def _load_session_with_timeout(ev, session_name: str, timeout: float = _SESSION_LOAD_TIMEOUT, **load_kwargs):
    """Load a FastF1 session with a wall-clock timeout.

    Returns the loaded session object, or raises ``FuturesTimeoutError`` /
    other exceptions on failure.
    """
    def _do_load():
        sess = ev.get_session(session_name)
        sess.load(**load_kwargs)
        return sess

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_do_load).result(timeout=timeout)


def get_session_times(ev, session_name: str) -> Optional[Tuple[datetime, datetime]]:
    if ev is None:
        return None
    try:
        sess = _load_session_with_timeout(
            ev, session_name,
            telemetry=False, laps=False, weather=False, messages=False,
        )
        start = _to_py_datetime(sess.session_info.get("StartDate"))
        end = _to_py_datetime(sess.session_info.get("EndDate"))
        if start and end:
            return start, end
        sched = getattr(ev, "session_info", None)
        if isinstance(sched, dict):
            s = _to_py_datetime(sched.get("StartDate"))
            e = _to_py_datetime(sched.get("EndDate"))
            if s and e:
                return s, e
        return None
    except FuturesTimeoutError:
        logger.warning("FastF1 session timing load timed out for %s", session_name)
        return None
    except Exception:
        return None


def get_session_classification(season: int, round_no: int, session_name: str):
    """Fetch session results/classification and apply patches/derivations if needed."""
    try:
        ev = get_event(season, round_no)
        if ev is None:
            return None
        sess = _load_session_with_timeout(
            ev, session_name,
            telemetry=False, laps=False, weather=False, messages=False,
        )

        # Try to get results from sess.results (often populated for finished sessions)
        results = getattr(sess, "results", None)

        # Fallback to get_classification() if results is empty
        if results is None or (hasattr(results, "empty") and results.empty):
            try:
                results = sess.get_classification()
            except Exception:
                results = None

        if results is not None and hasattr(results, "empty") and not results.empty:
            results = _patch_missing_positions(sess, results)
        elif results is not None and not hasattr(results, "empty"):
            # Results exists but might not be a DataFrame (unexpected but possible)
            pass
        elif results is not None and hasattr(results, "empty") and results.empty:
            # results exists but is empty DataFrame, still try to patch it via session object
            results = _patch_missing_positions(sess, results)

        return results
    except Exception:
        return None


def _patch_missing_positions(sess, results):
    """Work around FastF1 bug where laps 'Deleted' column contains None instead
    of bool, causing ``_calculate_quali_like_session_results`` to silently fail
    and leave Position/Q1/Q2/Q3 as NaN even though lap data exists.

    If positions are missing we coerce 'Deleted' to bool and re-trigger the
    internal classification calculation.  This is a no-op when FastF1 already
    computed positions correctly."""
    import pandas as pd

    pos_col = results.get("Position")
    if pos_col is not None and pd.to_numeric(pos_col, errors="coerce").notna().any():
        return results  # positions already populated, nothing to do

    try:
        # 1. Attempt internal FastF1 fix for qualifying sessions (Deleted-column bug)
        # Laps may not have been loaded yet — reload with laps=True so we can
        # fix the Deleted column and recompute the classification.
        def _reload_with_laps(s=sess):
            s.load(telemetry=False, laps=True, weather=False, messages=False)
            return s

        with ThreadPoolExecutor(max_workers=1) as pool:
            sess = pool.submit(_reload_with_laps).result(timeout=_SESSION_LOAD_TIMEOUT)

        laps = getattr(sess, "laps", None)

        if laps is not None and not laps.empty:
            if "Deleted" in laps.columns and laps["Deleted"].isna().any():
                laps["Deleted"] = laps["Deleted"].fillna(False).infer_objects(copy=False).astype(bool)
                sess._calculate_quali_like_session_results(force=True)
                results = getattr(sess, "results", results)

                # Check if fix worked
                pos_col = results.get("Position")
                if pos_col is not None and pd.to_numeric(pos_col, errors="coerce").notna().any():
                    logger.info("Patched FastF1 Deleted-column bug for %s; positions now available", sess.name)
                    return results

        # 2. Manual derivation as fallback (Q-times or Lap times)
        return patch_missing_positions(sess.name, results)

    except Exception as exc:
        logger.debug("_patch_missing_positions failed: %s", exc)

    return results


def patch_missing_positions(session_name: str, results: 'pd.DataFrame') -> 'pd.DataFrame':  # noqa: F821
    """Apply ranking derivation logic to a classification dataframe without needing a full session object."""
    import pandas as pd
    import numpy as np

    pos_col = results.get("Position")
    if pos_col is not None and pd.to_numeric(pos_col, errors="coerce").notna().any():
        return results

    try:
        results = results.copy()
        s_name = session_name.lower()
        # SQ and Shootout are also quali-style
        is_quali = any(x in s_name for x in ("qualifying", "shootout", "practice", "q", "sq"))

        derived = False
        if is_quali:
            # Sort: Q3 > Q2 > Q1. Lower time is better.
            q_cols = [c for c in ["Q3", "Q2", "Q1"] if c in results.columns]
            if q_cols:
                for c in q_cols:
                    # Clean up strings like 'NaT' or empty before conversion
                    if results[c].dtype == object:
                         results[c] = results[c].astype(str).str.strip().replace(['NaT', 'nan', ''], [np.nan, np.nan, np.nan])

                    if not pd.api.types.is_timedelta64_dtype(results[c]):
                        results[c] = pd.to_timedelta(results[c], errors="coerce")

                # We only derive if there's actual time data
                if results[q_cols].notna().any().any():
                    results = results.sort_values(by=q_cols, ascending=True, na_position='last')
                    results["Position"] = np.arange(1, len(results) + 1)
                    derived = True
        else:
            # Race/Sprint: Sort by Time (total) then BestLapTime. Lower is better.
            t_cols = [c for c in ["Time", "BestLapTime"] if c in results.columns]
            if t_cols:
                for c in t_cols:
                    if not pd.api.types.is_timedelta64_dtype(results[c]):
                        results[c] = pd.to_timedelta(results[c], errors="coerce")

                if results[t_cols].notna().any().any():
                    results = results.sort_values(by=t_cols, ascending=True, na_position='last')
                    results["Position"] = np.arange(1, len(results) + 1)
                    derived = True

        if derived:
            logger.info("Derived rankings for %s from available session times", session_name)
            return results

    except Exception as exc:
        logger.debug("_patch_missing_positions failed: %s", exc)

    return results


def get_session_weather_status(season: int, round_no: int, session_name: str) -> Optional[dict]:
    """Get weather conditions for a session including wet/dry status.
    
    Returns dict with:
      - is_wet: True if session had significant rain/wet conditions
      - rainfall: bool if rain was detected
      - track_status: 'WET' or 'DRY' based on conditions
    """
    try:
        ev = get_event(season, round_no)
        if ev is None:
            return None
        sess = _load_session_with_timeout(
            ev, session_name,
            telemetry=False, laps=True, weather=True, messages=False,
        )
        
        result = {"is_wet": False, "rainfall": False, "track_status": "DRY"}
        
        # Check weather data for rain
        weather_df = getattr(sess, "weather_data", None)
        if weather_df is not None and not weather_df.empty:
            if "Rainfall" in weather_df.columns:
                rainfall = weather_df["Rainfall"].any()
                result["rainfall"] = bool(rainfall)
                if rainfall:
                    result["is_wet"] = True
                    result["track_status"] = "WET"
        
        # Check laps for wet/inter compound usage
        laps = getattr(sess, "laps", None)
        if laps is not None and not laps.empty and "Compound" in laps.columns:
            compounds = set(laps["Compound"].dropna().unique())
            wet_compounds = {"WET", "INTERMEDIATE"}
            if compounds & wet_compounds:
                result["is_wet"] = True
                result["track_status"] = "WET"
        
        return result
    except Exception as e:
        logger.debug(f"Failed to get weather status: {e}")
        return None
