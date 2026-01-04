from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone

from .util import get_logger
from .data.jolpica import JolpicaClient

logger = get_logger(__name__)


def _entries_from_results(results: List[Dict]) -> List[Dict]:
    """Shape results rows into a deduplicated entry list."""
    entries: List[Dict] = []
    for r in results or []:
        drv = r.get("Driver", {}) if isinstance(r.get("Driver"), dict) else {}
        cons = r.get("Constructor", {}) if isinstance(r.get("Constructor"), dict) else {}
        entries.append({
            "driverId": drv.get("driverId"),
            "code": drv.get("code"),
            "givenName": drv.get("givenName"),
            "familyName": drv.get("familyName"),
            "permanentNumber": drv.get("permanentNumber"),
            "constructorId": cons.get("constructorId"),
            "constructorName": cons.get("name"),
        })
    # Deduplicate by driverId, preserving first
    seen = set()
    deduped = []
    for e in entries:
        did = e.get("driverId")
        if did and did not in seen:
            seen.add(did)
            deduped.append(e)
    return deduped


def _same_round_known_roster(jc: JolpicaClient, season: str, rnd: str) -> List[Dict]:
    """
    Try to get the event roster from the same round, in order:
      qualifying -> race -> sprint.
    """
    try:
        q = jc.get_qualifying_results(season, rnd)
        if q:
            return _entries_from_results(q)
    except Exception:
        pass
    try:
        r = jc.get_race_results(season, rnd)
        if r:
            return _entries_from_results(r)
    except Exception:
        pass
    try:
        s = jc.get_sprint_results(season, rnd)
        if s:
            return _entries_from_results(s)
    except Exception:
        pass
    return []


def _latest_completed_round_in_season(jc: JolpicaClient, season: str) -> Optional[str]:
    """Return latest round number in a season that has results (race preferred, else qualifying)."""
    try:
        races = jc.get_season_schedule(season)
        for r in reversed(races):
            rnd = str(r.get("round"))
            if jc.get_race_results(season, rnd):
                return rnd
            if jc.get_qualifying_results(season, rnd):
                return rnd
    except Exception:
        pass
    return None


def _previous_completed_round_in_season(jc: JolpicaClient, season: str, upto_round: str) -> Optional[str]:
    """Return the most recent completed round strictly before upto_round (race preferred, else qualifying)."""
    try:
        r_cur = int(upto_round)
    except Exception:
        return None
    for r in range(r_cur - 1, 0, -1):
        r_str = str(r)
        try:
            if jc.get_race_results(season, r_str):
                return r_str
            if jc.get_qualifying_results(season, r_str):
                return r_str
        except Exception:
            continue
    return None


def _previous_completed_event_global(jc: JolpicaClient, season: int, upto_round: Optional[int]) -> Optional[Tuple[str, str]]:
    """
    Find the immediately previous completed event going backwards:
      - If upto_round is given: search rounds < upto_round within this season.
      - Otherwise: latest completed round in this season.
      - If none: move to previous seasons, latest completed round.
    Returns (season, round) or None.
    """
    # First, within the same season
    if upto_round is not None:
        prev_in_season = _previous_completed_round_in_season(jc, str(season), str(upto_round))
        if prev_in_season:
            return str(season), prev_in_season
    latest_in_season = _latest_completed_round_in_season(jc, str(season))
    if latest_in_season:
        return str(season), latest_in_season

    # Then step into previous seasons
    for yr in range(season - 1, 1949, -1):
        latest_prev = _latest_completed_round_in_season(jc, str(yr))
        if latest_prev:
            return str(yr), latest_prev
    return None


def _roster_from_round(jc: JolpicaClient, season: str, rnd: str) -> List[Dict]:
    """
    Build roster from a specific (season, round), preferring race results, then qualifying.
    """
    try:
        r = jc.get_race_results(season, rnd)
        if r:
            return _entries_from_results(r)
    except Exception:
        pass
    try:
        q = jc.get_qualifying_results(season, rnd)
        if q:
            return _entries_from_results(q)
    except Exception:
        pass
    return []



def _roster_from_openf1(of1: Optional[Any], season: int, rnd: int) -> List[Dict]:
    """Attempt to fetch roster from OpenF1 API."""
    if of1 is None or not of1.enabled:
        return []
    
    # Try to find a relevant session (Practice 1 is usually first with entry list)
    # If we are verifying a future race, we want ANY session with data.
    # Force fresh check (no cache) to pick up late changes.
    session_key = of1.find_session(season, rnd, "Practice 1", use_cache=False)
    if not session_key:
        session_key = of1.find_session(season, rnd, "Qualifying", use_cache=False)
    
    if not session_key:
        return []

    try:
        drivers = of1.get_drivers(session_key, use_cache=False)
        if drivers.empty:
            return []
        
        roster = []
        for _, d in drivers.iterrows():
            # OpenF1 fields: driver_number, full_name, team_name, etc.
            roster.append({
                "driverId": d.get("name_acronym", "")[:3].lower(), # Fallback ID
                "code": d.get("name_acronym"),
                "givenName": d.get("first_name"),
                "familyName": d.get("last_name"),
                "permanentNumber": str(d.get("driver_number")),
                "constructorId": d.get("team_name", "").lower().replace(" ", "_"),
                "constructorName": d.get("team_name"),
            })
        logger.info(f"[roster] Fetched {len(roster)} drivers from OpenF1")
        return roster
    except Exception as e:
        logger.warning(f"[roster] OpenF1 fetch failed: {e}")
        return []


def _roster_from_fastf1(season: int, rnd: int) -> List[Dict]:
    """Attempt to fetch roster from FastF1 (live timing)."""
    try:
        # Import here to avoid circular dependencies if not already imported
        from .data import fastf1_backend as ff1
        
        ev = ff1.get_event(season, rnd)
        if ev is None:
            return []
            
        # Try to load session (e.g., FP1)
        # FastF1 often requires a session to be loaded to get drivers
        try:
            # We don't need full data, just metadata
            sess = ev.get_session("FP1")
            sess.load(telemetry=False, laps=False, weather=False, messages=False)
            results = sess.results
            if results is None or results.empty:
                return []
                
            roster = []
            for _, r in results.iterrows():
                roster.append({
                    "driverId": r["Abbreviation"].lower() if "Abbreviation" in r else None,
                    "code": r.get("Abbreviation"),
                    "givenName": r.get("FirstName"),
                    "familyName": r.get("LastName"),
                    "permanentNumber": str(r.get("DriverNumber")),
                    "constructorId": r.get("TeamName", "").lower().replace(" ", "_"),
                    "constructorName": r.get("TeamName"),
                })
            logger.info(f"[roster] Fetched {len(roster)} drivers from FastF1")
            return roster
        except Exception:
            # If FP1 fails, maybe it hasn't happened. FastF1 relies on finding data.
            pass
    except Exception as e:
        logger.warning(f"[roster] FastF1 fetch failed: {e}")
    return []


def derive_roster(
    jc: JolpicaClient,
    season: str,
    rnd: str,
    event_dt: Optional[datetime] = None,
    now_dt: Optional[datetime] = None,
    openf1_client: Optional[Any] = None 
) -> List[Dict]:
    """
    Derive roster using a prioritized cascade:
    1. OpenF1 (Session Entry List) - Best for verified entry lists slightly ahead of time.
    2. FastF1 (Live Timing) - Good for active weekends.
    3. Jolpica/Ergast (Results) - Reliable historical fallback.
    """
    # Normalise times
    if now_dt is None:
        now_dt = datetime.now(timezone.utc)
    if event_dt is not None and event_dt.tzinfo is None:
        event_dt = event_dt.replace(tzinfo=timezone.utc)

    s_int = int(season)
    r_int = int(rnd)

    # 1. OPTION A: OpenF1 (If available)
    # We check this even for future events if they are close enough to have an entry list
    roster = _roster_from_openf1(openf1_client, s_int, r_int)
    if roster:
        return roster

    # 2. OPTION B: Jolpica known results for this round
    # Check this BEFORE FastF1 because FastF1 may return test session data (e.g., post-season tests)
    same = _same_round_known_roster(jc, season, rnd)
    if same:
        return same

    # 3. OPTION C: FastF1 (live timing, good for active weekends before results are posted)
    roster = _roster_from_fastf1(s_int, r_int)
    if roster:
        return roster

    # 4. OPTION D: Previous completed event (fallback for future events)
    try:
        prev = _previous_completed_event_global(jc, s_int, r_int)
    except Exception:
        prev = _previous_completed_event_global(jc, s_int, None)
    
    if prev:
        ps, pr = prev
        logger.info(f"[roster] Falling back to roster from {ps} R{pr}")
        return _roster_from_round(jc, ps, pr)

    return []