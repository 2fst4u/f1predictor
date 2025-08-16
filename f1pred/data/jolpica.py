from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import time

from ..util import session_with_retries, http_get_json, get_logger

logger = get_logger(__name__)


class JolpicaClient:
    """
    Thin client for Jolpica's Ergast-compatible API (pure data access).
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
        return mr.get("RaceTable", {}).get("Races", []) or []

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
        return (races[0].get("Results", []) if races else []) or []

    def get_qualifying_results(self, season: str, rnd: str) -> List[Dict[str, Any]]:
        js = self._get(f"{season}/{rnd}/qualifying.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        races = mr.get("RaceTable", {}).get("Races", [])
        return (races[0].get("QualifyingResults", []) if races else []) or []

    def get_sprint_results(self, season: str, rnd: str) -> List[Dict[str, Any]]:
        js = self._get(f"{season}/{rnd}/sprint.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        races = mr.get("RaceTable", {}).get("Races", [])
        return (races[0].get("SprintResults", []) if races else []) or []

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