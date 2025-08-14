from __future__ import annotations
from typing import Optional, Tuple, Any
from datetime import datetime, timezone

try:
    import fastf1  # type: ignore
except Exception:  # pragma: no cover
    fastf1 = None  # type: ignore

import logging

logger = logging.getLogger(__name__)

_CACHE_INITED = False


def init_fastf1(cache_dir: str) -> bool:
    """
    Enable FastF1 on-disk cache. Returns True if cache initialised or already active.
    No-op (False) if FastF1 is not installed.
    """
    global _CACHE_INITED
    if fastf1 is None:
        logger.warning("FastF1 is not installed; fastf1_backend is disabled.")
        return False
    if _CACHE_INITED:
        return True
    try:
        fastf1.Cache.enable_cache(cache_dir)
        _CACHE_INITED = True
        return True
    except Exception as e:
        logger.warning(f"FastF1 cache init failed: {e}")
        return False


def _to_py_datetime(dt_obj: Any) -> Optional[datetime]:
    """
    Coerce FastF1 time-like object (pandas.Timestamp or datetime) to a Python datetime in UTC.
    Returns None if conversion fails.
    """
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
    """
    Return a FastF1 Event object, or None if unavailable.
    """
    if fastf1 is None:
        logger.info("FastF1 unavailable; get_event returning None.")
        return None
    try:
        ev = fastf1.get_event(season, round_no)
        return ev
    except Exception as e:
        logger.info(f"FastF1 get_event failed for {season} R{round_no}: {e}")
        return None


def get_session_times(ev, session_name: str) -> Optional[Tuple[datetime, datetime]]:
    """
    Load a session and return (start, end) datetimes in UTC (tz-aware).
    Returns None if unavailable.
    """
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
    except Exception as e:
        ev_name = getattr(ev, 'EventName', None) or getattr(ev, 'OfficialEventName', 'Unknown')
        logger.info(f"FastF1 get_session_times failed for {ev_name} {session_name}: {e}")
        return None


def get_session_classification(season: int, round_no: int, session_name: str):
    """
    Return a classification DataFrame for a session if available, else None.
    Columns typically include: Position, DriverNumber, Abbreviation, etc.
    """
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
            cls = sess.get_classification()
            return cls
        except Exception:
            return None
    except Exception:
        return None