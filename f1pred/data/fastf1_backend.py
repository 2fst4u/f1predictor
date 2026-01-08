from __future__ import annotations
from typing import Optional, Tuple, Any
from datetime import datetime, timezone
import logging

try:
    import fastf1  # type: ignore
except Exception:  # pragma: no cover
    fastf1 = None  # type: ignore

logger = logging.getLogger(__name__)
_CACHE_INITED = False


def init_fastf1(cache_dir: str) -> bool:
    """Enable cache and silence FastF1 noisy logs; never raise."""
    global _CACHE_INITED
    if fastf1 is None:
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
    if fastf1 is None:
        return None
    try:
        return fastf1.get_event(season, round_no)
    except Exception:
        return None


def get_session_times(ev, session_name: str) -> Optional[Tuple[datetime, datetime]]:
    if fastf1 is None or ev is None:
        return None
    try:
        sess = ev.get_session(session_name)
        sess.load(telemetry=False, laps=False, weather=False, messages=False)
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
    except Exception:
        return None


def get_session_classification(season: int, round_no: int, session_name: str):
    if fastf1 is None:
        return None
    try:
        ev = get_event(season, round_no)
        if ev is None:
            return None
        sess = ev.get_session(session_name)
        sess.load(telemetry=False, laps=False, weather=False, messages=False)
        results = getattr(sess, "results", None)
        if results is not None and hasattr(results, "empty") and not results.empty:
            return results
        try:
            return sess.get_classification()
        except Exception:
            return None
    except Exception:
        return None


def get_session_weather_status(season: int, round_no: int, session_name: str) -> Optional[dict]:
    """Get weather conditions for a session including wet/dry status.
    
    Returns dict with:
      - is_wet: True if session had significant rain/wet conditions
      - rainfall: bool if rain was detected
      - track_status: 'WET' or 'DRY' based on conditions
    """
    if fastf1 is None:
        return None
    try:
        ev = get_event(season, round_no)
        if ev is None:
            return None
        sess = ev.get_session(session_name)
        sess.load(telemetry=False, laps=True, weather=True, messages=False)
        
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