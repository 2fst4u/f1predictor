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
            if not r.get("round"):
                continue
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
    # Optimization: Check global "latest" pointer to skip scanning future seasons/rounds
    try:
        last_s_str, last_r_str = jc.get_latest_season_and_round()
        last_s, last_r = int(last_s_str), int(last_r_str)

        # Case 1: Requested season is in the future relative to latest completed event
        if season > last_s:
            return last_s_str, last_r_str

        # Case 2: Requested season IS the latest season
        if season == last_s:
            # If we want the latest in this season, or if our cutoff is beyond the latest
            if upto_round is None or upto_round > last_r:
                return last_s_str, last_r_str
            # If upto_round <= last_r, we must search backwards as usual (might be gaps?)
    except Exception as e:
        logger.info(f"[roster] Optimization failed: {e}; falling back to full scan")

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


def _get_canonical_mapping(jc: JolpicaClient, season: str) -> Dict[str, Any]:
    """Build lookup tables to map various identities to canonical Jolpica/Ergast IDs."""
    try:
        raw_entries = jc.get_season_entry_list(season)
        if not raw_entries:
            return {}

        driver_map = {}
        constructor_map = {}

        for r in raw_entries:
            drv = r.get("Driver", {})
            cons = r.get("Constructor", {})

            d_id = drv.get("driverId")
            c_id = cons.get("constructorId")
            c_name = cons.get("name")

            if d_id:
                # Map by Number
                num = drv.get("permanentNumber")
                if num:
                    driver_map[str(num)] = d_id

                # Map by Code (Abbreviation)
                code = drv.get("code")
                if code:
                    driver_map[code.upper()] = d_id

                # Map by Name (Normalised)
                gn = drv.get("givenName", "")
                fn = drv.get("familyName", "")
                if gn and fn:
                    name_key = f"{gn} {fn}".lower().strip()
                    driver_map[name_key] = d_id

            if c_id and c_name:
                # Map by Name (Normalised)
                constructor_map[c_name.lower().strip()] = c_id

        return {
            "drivers": driver_map,
            "constructors": constructor_map
        }
    except Exception as e:
        logger.warning(f"[roster] Failed to build canonical mapping: {e}")
        return {}


def _roster_from_fastf1(season: int, rnd: int, mapping: Optional[Dict] = None) -> List[Dict]:
    """Attempt to fetch roster from FastF1 (live timing)."""
    try:
        # Import here to avoid circular dependencies if not already imported
        from .data import fastf1_backend as ff1

        ev = ff1.get_event(season, rnd)
        if ev is None:
            return []

        # Try to load a session to get the roster.
        # We try multiple sessions in case some data is missing or hasn't occurred yet.
        # Order: FP1 (standard), then sessions that might have occurred.
        session_names = ["FP1", "Sprint Qualifying", "Sprint Shootout", "Qualifying", "Sprint", "Race"]
        results = None

        for sname in session_names:
            try:
                sess = ev.get_session(sname)
                sess.load(telemetry=False, laps=False, weather=False, messages=False)
                if sess.results is not None and not sess.results.empty:
                    results = sess.results
                    logger.info(f"[roster] Found roster in session: {sname}")
                    break
            except Exception:
                continue

        if results is None or results.empty:
            return []

        roster = []
        d_map = (mapping or {}).get("drivers", {})
        c_map = (mapping or {}).get("constructors", {})

        for _, r in results.iterrows():
            # Try to canonicalize driverId
            abbr = r.get("Abbreviation")
            num = r.get("DriverNumber")
            gn = r.get("FirstName")
            fn = r.get("LastName")

            did = None
            if abbr and abbr.upper() in d_map:
                did = d_map[abbr.upper()]
            elif num and str(num) in d_map:
                did = d_map[str(num)]
            elif gn and fn:
                name_key = f"{gn} {fn}".lower().strip()
                if name_key in d_map:
                    did = d_map[name_key]

            # Fallback to abbreviation if not mapped
            if not did:
                did = abbr.lower() if abbr else None

            # Try to canonicalize constructorId
            tname = r.get("TeamName")
            cid = None
            if tname and tname.lower().strip() in c_map:
                cid = c_map[tname.lower().strip()]

            # Fallback to normalised TeamName
            if not cid:
                cid = tname.lower().replace(" ", "_") if tname else None

            roster.append({
                "driverId": did,
                "code": abbr,
                "givenName": gn,
                "familyName": fn,
                "permanentNumber": str(num) if num else None,
                "constructorId": cid,
                "constructorName": tname,
            })

        if roster:
            logger.info(f"[roster] Fetched {len(roster)} drivers from FastF1")
            return roster
    except Exception as e:
        logger.warning(f"[roster] FastF1 fetch failed: {e}")
    return []


def derive_roster(
    jc: JolpicaClient,
    season: str,
    rnd: str,
    event_dt: Optional[datetime] = None,
    now_dt: Optional[datetime] = None,
) -> List[Dict]:
    """
    Derive roster using a prioritized cascade:
    1. FastF1 (Live Timing) - Good for active weekends.
    2. Jolpica/Ergast (Results) - Reliable historical fallback.
    """
    # Normalise times
    if now_dt is None:
        now_dt = datetime.now(timezone.utc)
    if event_dt is not None and event_dt.tzinfo is None:
        event_dt = event_dt.replace(tzinfo=timezone.utc)

    s_int = int(season)
    r_int = int(rnd)

    # 1. OPTION A: Jolpica known results for this round
    # Check this BEFORE FastF1 because FastF1 may return test session data (e.g., post-season tests)
    same = _same_round_known_roster(jc, season, rnd)
    # Check if Jolpica results are reasonably complete (at least 20 drivers for a modern season)
    if same and (len(same) >= 20 or int(season) < 2010):
        return same

    # 2. OPTION B: FastF1 (live timing, good for active weekends before results are posted)
    # Fetch mapping for canonicalization
    mapping = _get_canonical_mapping(jc, season)
    roster = _roster_from_fastf1(s_int, r_int, mapping=mapping)
    if roster:
        return roster

    # 3. OPTION C: Season entry list (for start of season when no races have happened yet)
    # If the requested season matches the current one (or future), try to get the official entry list
    # This prevents falling back to the previous season's roster (e.g. 2025) at the start of 2026.
    raw_entries = jc.get_season_entry_list(season)
    if raw_entries:
        logger.info(f"[roster] Using official entry list for Season {season}")
        return _entries_from_results(raw_entries)

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
