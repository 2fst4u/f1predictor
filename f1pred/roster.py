from __future__ import annotations
from typing import List, Dict, Optional, Tuple

from .util import get_logger
from .data.jolpica import JolpicaClient

logger = get_logger(__name__)


def _entries_from_results(results: List[Dict]) -> List[Dict]:
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
    # Deduplicate by driverId, preserving first occurrence
    seen = set()
    deduped = []
    for e in entries:
        did = e.get("driverId")
        if did and did not in seen:
            seen.add(did)
            deduped.append(e)
    return deduped


def _previous_completed_round_in_season(jc: JolpicaClient, season: str, upto_round: str) -> Optional[str]:
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


def _latest_completed_round_in_season(jc: JolpicaClient, season: str) -> Optional[str]:
    try:
        races = jc.get_season_schedule(season)
        for r in reversed(races):
            r_str = str(r.get("round"))
            if jc.get_race_results(season, r_str):
                return r_str
            if jc.get_qualifying_results(season, r_str):
                return r_str
    except Exception:
        pass
    return None


def _latest_completed_round_global(jc: JolpicaClient, start_season: int) -> Optional[Tuple[str, str]]:
    for yr in range(start_season, 1949, -1):
        r = _latest_completed_round_in_season(jc, str(yr))
        if r:
            return str(yr), r
    return None


def derive_roster(
    jc: JolpicaClient,
    season: str,
    rnd: str,
    prefer_same_round_qualifying: bool = True
) -> List[Dict]:
    """
    Derive event roster using one consistent logic:
      1) If same-round qualifying exists, use it (best for same-weekend Saturday).
      2) Else previous completed round in same season (race results first; else qualifying).
      3) Else season drivers mapped via standings (constructor inferred from standings).
      4) Else latest completed round globally (prior seasons; race > qualifying).

    Returns a list of dicts with:
      driverId, code, givenName, familyName, permanentNumber, constructorId, constructorName
    """
    # 1) Same-round qualifying (best when schedule is imminent)
    if prefer_same_round_qualifying:
        try:
            q_cur = jc.get_qualifying_results(season, rnd)
            if q_cur:
                return _entries_from_results(q_cur)
        except Exception:
            pass

    # 2) Prior completed round in same season
    prev_rnd = _previous_completed_round_in_season(jc, season, rnd)
    if prev_rnd:
        try:
            res = jc.get_race_results(season, prev_rnd)
            if not res:
                res = jc.get_qualifying_results(season, prev_rnd)
            if res:
                return _entries_from_results(res)
        except Exception as e:
            logger.info(f"derive_roster: failed prior round {season} R{prev_rnd}: {e}")

    # 3) Season drivers + standings-based constructor mapping
    try:
        drivers = jc.get_drivers_for_season(season)
    except Exception as e:
        logger.info(f"derive_roster: get_drivers_for_season failed for {season}: {e}")
        drivers = []

    constr_map: Dict[str, Optional[str]] = {}
    try:
        standings = jc.get_standings(season).get("StandingsLists", [])
        if standings:
            for dr in standings[0].get("DriverStandings", []):
                drv = dr.get("Driver", {}) or {}
                constructors = dr.get("Constructors", []) or []
                if constructors:
                    constr_map[drv.get("driverId")] = constructors[0].get("constructorId")
    except Exception as e:
        logger.info(f"derive_roster: get_standings failed for {season}: {e}")

    if drivers:
        entries = []
        for d in drivers:
            entries.append({
                "driverId": d.get("driverId"),
                "code": d.get("code"),
                "givenName": d.get("givenName"),
                "familyName": d.get("familyName"),
                "permanentNumber": d.get("permanentNumber"),
                "constructorId": constr_map.get(d.get("driverId")),
                "constructorName": None
            })
        return entries

    # 4) Prior seasons (global latest completed round)
    try:
        gp = _latest_completed_round_global(jc, int(season) - 1)
        if gp:
            g_season, g_round = gp
            res = jc.get_race_results(g_season, g_round)
            if not res:
                res = jc.get_qualifying_results(g_season, g_round)
            if res:
                return _entries_from_results(res)
    except Exception as e:
        logger.info(f"derive_roster: global fallback failed from season {season}: {e}")

    # If everything fails, return empty list; caller should handle gracefully
    return []