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


def derive_roster(
    jc: JolpicaClient,
    season: str,
    rnd: str,
    event_dt: Optional[datetime] = None,
    now_dt: Optional[datetime] = None
) -> List[Dict]:
    """
    Single source of truth for roster derivation with your specified logic:

    - If prediction is in the past (event_dt < now): use the known roster for that event
      (same round), trying qualifying -> race -> sprint.

    - If prediction is in the future: attempt a known roster (same round). If that fails,
      use the roster from the previous event (race or qualifying, whichever happened just
      prior). The "previous event" search walks back through rounds in this season, then
      previous seasons if needed.

    Returns a list of dicts with:
      driverId, code, givenName, familyName, permanentNumber, constructorId, constructorName
    """
    # Normalise times
    if now_dt is None:
        now_dt = datetime.now(timezone.utc)
    if event_dt is not None and event_dt.tzinfo is None:
        event_dt = event_dt.replace(tzinfo=timezone.utc)

    # 1) Past event: use known roster for this round if available
    if event_dt is not None and event_dt < now_dt:
        same = _same_round_known_roster(jc, season, rnd)
        if same:
            return same
        # Fall back to previous event if same round has no posted results
        try:
            prev = _previous_completed_event_global(jc, int(season), int(rnd))
        except Exception:
            prev = _previous_completed_event_global(jc, int(season), None)
        if prev:
            ps, pr = prev
            return _roster_from_round(jc, ps, pr)
        return []

    # 2) Future event: try same-round known roster (e.g., if qualifying already happened)
    same = _same_round_known_roster(jc, season, rnd)
    if same:
        return same

    # Else use previous completed event (race > qualifying), walking back across seasons if needed
    try:
        prev = _previous_completed_event_global(jc, int(season), int(rnd))
    except Exception:
        prev = _previous_completed_event_global(jc, int(season), None)
    if prev:
        ps, pr = prev
        return _roster_from_round(jc, ps, pr)

    return []