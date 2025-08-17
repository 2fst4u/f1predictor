from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import time
import random

import requests
from requests import HTTPError

from ..util import session_with_retries, http_get_json, get_logger

logger = get_logger(__name__)


class JolpicaClient:
    """
    Thin client for Jolpica's Ergast-compatible API (pure data access).

    Changes:
    - Adds season-bulk endpoints to reduce per-round calls.
    - Adds exponential backoff for HTTP 429 responses on all requests.
    """

    def __init__(self, base_url: str, timeout: int = 30, rate_limit_sleep: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limit_sleep = rate_limit_sleep
        self.session = session_with_retries()

    def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        max_429_retries: int = 6,
        backoff_base: float = 0.5,
        backoff_max: float = 8.0,
    ) -> Dict[str, Any]:
        """
        GET with exponential backoff on 429, and small inter-call sleep (if configured).
        backoff schedule ~ base * 2^attempt, capped at backoff_max, with jitter.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        attempt = 0
        while True:
            try:
                data = http_get_json(self.session, url, params=params or {}, timeout=self.timeout)
                if self.rate_limit_sleep and self.rate_limit_sleep > 0:
                    time.sleep(self.rate_limit_sleep)
                return data
            except Exception as e:
                status_code = None
                if isinstance(e, HTTPError) and getattr(e, "response", None) is not None:
                    status_code = getattr(e.response, "status_code", None)
                if status_code is None:
                    status_code = getattr(e, "status_code", None) or getattr(e, "status", None)
                if status_code == 429 and attempt < max_429_retries:
                    sleep_s = min(backoff_max, backoff_base * (2 ** attempt))
                    jitter = sleep_s * (0.8 + 0.4 * random.random())
                    logger.info(f"429 from {url}; backing off {jitter:.2f}s (attempt {attempt + 1}/{max_429_retries})")
                    time.sleep(jitter)
                    attempt += 1
                    continue
                raise

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

    # Round-level results

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

    # Drivers/constructors/standings

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

    # Bulk season endpoints (to avoid per-round 429s and reduce calls)

    def get_season_race_results(self, season: str) -> List[Dict[str, Any]]:
        """All race results for a season across all rounds."""
        js = self._get(f"{season}/results.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        return mr.get("RaceTable", {}).get("Races", []) or []

    def get_season_qualifying_results(self, season: str) -> List[Dict[str, Any]]:
        """All qualifying results for a season across all rounds."""
        js = self._get(f"{season}/qualifying.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        return mr.get("RaceTable", {}).get("Races", []) or []

    def get_season_sprint_results(self, season: str) -> List[Dict[str, Any]]:
        """All sprint results for a season across all rounds."""
        js = self._get(f"{season}/sprint.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        return mr.get("RaceTable", {}).get("Races", []) or []