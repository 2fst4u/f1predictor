from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time

from ..util import session_with_retries, http_get_json, get_logger

logger = get_logger(__name__)


class JolpicaClient:
    """
    Thin client for Jolpica's Ergast-compatible API.
    """

    def __init__(self, base_url: str, timeout: int = 30, rate_limit_sleep: float = 0.5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limit_sleep = rate_limit_sleep
        self.session = session_with_retries()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        data = http_get_json(self.session, url, params=params or {}, timeout=self.timeout)
        if self.rate_limit_sleep and self.rate_limit_sleep > 0:
            time.sleep(self.rate_limit_sleep)
        return data

    @staticmethod
    def _extract_mrdata(data: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(data, dict) and "MRData" in data:
            return data["MRData"]
        return data

    # Schedules and events

    def get_season_schedule(self, season: str) -> List[Dict[str, Any]]:
        js = self._get(f"{season}.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        races = mr.get("RaceTable", {}).get("Races", [])
        return races or []

    def get_event(self, season: str, rnd: str) -> Optional[Dict[str, Any]]:
        try:
            races = self.get_season_schedule(season)
            for r in races:
                if str(r.get("round")) == str(rnd):
                    return r
        except Exception as e:
            logger.info(f"get_event failed for {season} R{rnd}: {e}")
        return None

    # Locators

    def get_latest_season_and_round(self) -> Tuple[str, str]:
        js = self._get("current/last.json")
        mr = self._extract_mrdata(js)
        race = (mr.get("RaceTable", {}) or {}).get("Races", [{}])[0]
        return str(race.get("season")), str(race.get("round"))

    def get_next_round(self) -> Tuple[str, str]:
        js = self._get("current/next.json")
        mr = self._extract_mrdata(js)
        race = (mr.get("RaceTable", {}) or {}).get("Races", [{}])[0]
        return str(race.get("season")), str(race.get("round"))

    # Results/standings

    def get_race_results(self, season: str, rnd: str) -> List[Dict[str, Any]]:
        js = self._get(f"{season}/{rnd}/results.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        races = mr.get("RaceTable", {}).get("Races", [])
        if not races:
            return []
        return races[0].get("Results", []) or []

    def get_qualifying_results(self, season: str, rnd: str) -> List[Dict[str, Any]]:
        js = self._get(f"{season}/{rnd}/qualifying.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        races = mr.get("RaceTable", {}).get("Races", [])
        if not races:
            return []
        return races[0].get("QualifyingResults", []) or []

    def get_sprint_results(self, season: str, rnd: str) -> List[Dict[str, Any]]:
        js = self._get(f"{season}/{rnd}/sprint.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        races = mr.get("RaceTable", {}).get("Races", [])
        if not races:
            return []
        return races[0].get("SprintResults", []) or []

    def get_drivers_for_season(self, season: str) -> List[Dict[str, Any]]:
        js = self._get(f"{season}/drivers.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        return mr.get("DriverTable", {}).get("Drivers", []) or []

    def get_constructors_for_season(self, season: str) -> List[Dict[str, Any]]:
        js = self._get(f"{season}/constructors.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        return mr.get("ConstructorTable", {}).get("Constructors", []) or []

    def get_standings(self, season: str) -> Dict[str, Any]:
        js = self._get(f"{season}/driverStandings.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        return mr.get("StandingsTable", {}) or {}

    # Helpers for roster inference

    def _entries_from_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
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
        return entries

    def get_previous_completed_round_in_season(self, season: str, upto_round: str) -> Optional[str]:
        try:
            r_cur = int(upto_round)
        except Exception:
            return None
        for r in range(r_cur - 1, 0, -1):
            r_str = str(r)
            try:
                if self.get_race_results(season, r_str):
                    return r_str
                if self.get_qualifying_results(season, r_str):
                    return r_str
            except Exception:
                continue
        return None

    # Roster inference

    def derive_entry_list(self, season: str, rnd: str, schedule_races: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Infer an entry list for a given season/round.
        Strategy:
          1) Use the most recent completed round in the SAME season (race results preferred, else qualifying).
          2) If none exist (e.g., round 1), fall back to season drivers and map teams via standings (if available).
          3) As last resort, use current-last event entries (may cross seasons).
        """
        prev_round = self.get_previous_completed_round_in_season(season, rnd)
        if prev_round:
            try:
                results = self.get_race_results(season, prev_round)
                if not results:
                    results = self.get_qualifying_results(season, prev_round)
                entries = self._entries_from_results(results)
                if entries:
                    seen = set()
                    deduped = []
                    for e in entries:
                        did = e.get("driverId")
                        if did and did not in seen:
                            seen.add(did)
                            deduped.append(e)
                    return deduped
            except Exception as e:
                logger.info(f"derive_entry_list: failed prior round {season} R{prev_round}: {e}")

        drivers = []
        try:
            drivers = self.get_drivers_for_season(season)
        except Exception as e:
            logger.info(f"derive_entry_list: get_drivers_for_season failed for {season}: {e}")

        constr_map: Dict[str, Optional[str]] = {}
        try:
            standings = self.get_standings(season).get("StandingsLists", [])
            if standings:
                for dr in standings[0].get("DriverStandings", []):
                    drv = dr.get("Driver", {}) or {}
                    constructors = dr.get("Constructors", []) or []
                    if constructors:
                        constr_map[drv.get("driverId")] = constructors[0].get("constructorId")
        except Exception as e:
            logger.info(f"derive_entry_list: get_standings failed for {season}: {e}")

        entries_fallback: List[Dict[str, Any]] = []
        for d in drivers or []:
            entries_fallback.append({
                "driverId": d.get("driverId"),
                "code": d.get("code"),
                "givenName": d.get("givenName"),
                "familyName": d.get("familyName"),
                "permanentNumber": d.get("permanentNumber"),
                "constructorId": constr_map.get(d.get("driverId")),
                "constructorName": None
            })

        if not entries_fallback:
            try:
                latest_season, latest_round = self.get_latest_season_and_round()
                results = self.get_race_results(latest_season, latest_round)
                entries = self._entries_from_results(results)
                if entries:
                    return entries
            except Exception:
                pass

        return entries_fallback