from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import time
import random
from email.utils import parsedate_to_datetime

import requests
from requests import HTTPError

from ..util import session_with_retries, http_get_json, get_logger

logger = get_logger(__name__)


class JolpicaClient:
    """
    Thin client for Jolpica's Ergast-compatible API (pure data access).

    - Season-bulk endpoints reduce per-round calls.
    - Exponential backoff with Retry-After support for HTTP 429.
    """

    # Sentinel: Limit pagination to prevent DoS via infinite loops or huge datasets
    MAX_PAGINATION_PAGES = 20

    def __init__(self, base_url: str, timeout: int = 30, rate_limit_sleep: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limit_sleep = rate_limit_sleep
        self.session = session_with_retries()

    @staticmethod
    def _retry_after_seconds(headers: Dict[str, str], attempt: int, base: float, cap: float) -> float:
        """
        Choose backoff duration:
          - If Retry-After is a number (seconds), use it.
          - If Retry-After is an HTTP-date, compute seconds until then (min 0).
          - Else use exponential backoff: base * 2^attempt, capped at cap, with jitter.
        """
        ra = headers.get("Retry-After")
        if ra:
            # Hard cap for Retry-After to prevent DoS via infinite/long sleep (max 5 minutes)
            MAX_RETRY_AFTER = 300.0
            # Numeric seconds
            try:
                secs = float(ra)
                return min(max(0.0, secs), MAX_RETRY_AFTER)
            except Exception:
                pass
            # HTTP-date
            try:
                dt = parsedate_to_datetime(ra)
                if dt:
                    now_epoch = time.time()
                    diff = dt.timestamp() - now_epoch
                    return min(max(0.0, diff), MAX_RETRY_AFTER)
            except Exception:
                pass
        # Fallback: exponential with jitter
        backoff = min(cap, base * (2 ** attempt))
        jitter = backoff * (0.8 + 0.4 * random.random())
        return jitter

    def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        max_429_retries: int = 6,
        backoff_base: float = 0.5,
        backoff_max: float = 30.0,
    ) -> Dict[str, Any]:
        """
        GET with Retry-After-aware backoff on 429. Logs chosen backoff and outcome.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        attempt = 0
        slept_total = 0.0

        while True:
            try:
                data = http_get_json(self.session, url, params=params or {}, timeout=self.timeout)
                if self.rate_limit_sleep and self.rate_limit_sleep > 0:
                    time.sleep(self.rate_limit_sleep)
                if attempt > 0:
                    logger.info(
                        f"429 retries succeeded for {url} after {attempt} attempt(s), slept {slept_total:.2f}s total"
                    )
                return data
            except HTTPError as e:
                status_code = getattr(e.response, "status_code", None) if e.response is not None else None
                if status_code == 429 and attempt < max_429_retries:
                    headers = e.response.headers if e.response is not None else {}
                    sleep_s = self._retry_after_seconds(headers, attempt, backoff_base, backoff_max)
                    logger.info(
                        f"429 from {url}; backing off {sleep_s:.2f}s (attempt {attempt + 1}/{max_429_retries})"
                    )
                    time.sleep(sleep_s)
                    slept_total += sleep_s
                    attempt += 1
                    continue
                raise
            except Exception:
                raise

    @staticmethod
    def _extract_mrdata(data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            # If data is not a dict (e.g. raw string from http_get_json fallback),
            # return empty dict to prevent AttributeError downstream.
            logger.warning(f"Unexpected data format from Jolpica API: {type(data)}")
            return {}

        if "MRData" in data:
            return data["MRData"]

        return data

    # Input validation

    def _validate_season(self, season: str) -> str:
        """Ensure season is a valid year or 'current'."""
        s = str(season).strip()
        if s == "current":
            return s
        if not s.isdigit() or len(s) != 4:
            # Use repr() to prevent log injection from malicious input containing newlines
            raise ValueError(f"Invalid season: {repr(season)}")
        return s

    def _validate_round(self, rnd: str) -> str:
        """Ensure round is a valid number or 'last'/'next'."""
        r = str(rnd).strip()
        if r in ("last", "next"):
            return r

        # Enforce length limit (F1 rounds are rarely > 25, 5 digits is extremely generous)
        if len(r) > 5:
            raise ValueError(f"Invalid round (too long): {repr(r[:10])}...")

        if not r.isdigit():
            # Use repr() to prevent log injection from malicious input
            raise ValueError(f"Invalid round: {repr(rnd)}")
        return r

    # Schedules and events

    def get_season_schedule(self, season: str) -> List[Dict[str, Any]]:
        season = self._validate_season(season)
        js = self._get(f"{season}.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        return mr.get("RaceTable", {}).get("Races", []) or []

    def get_event(self, season: str, rnd: str) -> Optional[Dict[str, Any]]:
        try:
            # We don't validate round here strictly because we iterate over schedule,
            # but validating season is important.
            season = self._validate_season(season)
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
        season = self._validate_season(season)
        rnd = self._validate_round(rnd)
        js = self._get(f"{season}/{rnd}/results.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        races = mr.get("RaceTable", {}).get("Races", [])
        return (races[0].get("Results", []) if races else []) or []

    def get_qualifying_results(self, season: str, rnd: str) -> List[Dict[str, Any]]:
        season = self._validate_season(season)
        rnd = self._validate_round(rnd)
        js = self._get(f"{season}/{rnd}/qualifying.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        races = mr.get("RaceTable", {}).get("Races", [])
        return (races[0].get("QualifyingResults", []) if races else []) or []

    def get_sprint_results(self, season: str, rnd: str) -> List[Dict[str, Any]]:
        season = self._validate_season(season)
        rnd = self._validate_round(rnd)
        js = self._get(f"{season}/{rnd}/sprint.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        races = mr.get("RaceTable", {}).get("Races", [])
        return (races[0].get("SprintResults", []) if races else []) or []

    # Drivers/constructors/standings

    def get_drivers_for_season(self, season: str) -> List[Dict[str, Any]]:
        season = self._validate_season(season)
        js = self._get(f"{season}/drivers.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        return mr.get("DriverTable", {}).get("Drivers", []) or []

    def get_constructors_for_season(self, season: str) -> List[Dict[str, Any]]:
        season = self._validate_season(season)
        js = self._get(f"{season}/constructors.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        return mr.get("ConstructorTable", {}).get("Constructors", []) or []

    def get_standings(self, season: str) -> Dict[str, Any]:
        season = self._validate_season(season)
        js = self._get(f"{season}/driverStandings.json", params={"limit": 1000})
        mr = self._extract_mrdata(js)
        return mr.get("StandingsTable", {}) or {}

    # Bulk season endpoints (reduce calls and 429s)

    def get_season_race_results(self, season: str) -> List[Dict[str, Any]]:
        """All race results for a season across all rounds (paginated)."""
        season = self._validate_season(season)
        all_races: Dict[str, Dict[str, Any]] = {}  # round -> race dict
        offset = 0
        limit = 100  # API-enforced limit
        pages_fetched = 0
        
        while True:
            if pages_fetched >= self.MAX_PAGINATION_PAGES:
                logger.warning(f"JolpicaClient: race results pagination limit ({self.MAX_PAGINATION_PAGES}) reached for {season}")
                break

            js = self._get(f"{season}/results.json", params={"limit": limit, "offset": offset})
            mr = self._extract_mrdata(js)
            races = mr.get("RaceTable", {}).get("Races", []) or []
            pages_fetched += 1
            
            # Merge results into existing race dicts (API may split results across pages)
            for r in races:
                rnd = str(r.get("round"))
                if rnd not in all_races:
                    all_races[rnd] = r
                else:
                    # Extend results for this round
                    existing = all_races[rnd].get("Results", []) or []
                    new_results = r.get("Results", []) or []
                    all_races[rnd]["Results"] = existing + new_results
            
            total = int(mr.get("total", 0))
            offset += limit
            if offset >= total or not races:
                break
        
        return list(all_races.values())

    def get_season_qualifying_results(self, season: str) -> List[Dict[str, Any]]:
        """All qualifying results for a season across all rounds (paginated)."""
        season = self._validate_season(season)
        all_races: Dict[str, Dict[str, Any]] = {}
        offset = 0
        limit = 100
        pages_fetched = 0
        
        while True:
            if pages_fetched >= self.MAX_PAGINATION_PAGES:
                logger.warning(f"JolpicaClient: qualifying results pagination limit ({self.MAX_PAGINATION_PAGES}) reached for {season}")
                break

            js = self._get(f"{season}/qualifying.json", params={"limit": limit, "offset": offset})
            mr = self._extract_mrdata(js)
            races = mr.get("RaceTable", {}).get("Races", []) or []
            pages_fetched += 1
            
            for r in races:
                rnd = str(r.get("round"))
                if rnd not in all_races:
                    all_races[rnd] = r
                else:
                    existing = all_races[rnd].get("QualifyingResults", []) or []
                    new_results = r.get("QualifyingResults", []) or []
                    all_races[rnd]["QualifyingResults"] = existing + new_results
            
            total = int(mr.get("total", 0))
            offset += limit
            if offset >= total or not races:
                break
        
        return list(all_races.values())

    def get_season_sprint_results(self, season: str) -> List[Dict[str, Any]]:
        """All sprint results for a season across all rounds (paginated)."""
        season = self._validate_season(season)
        all_races: Dict[str, Dict[str, Any]] = {}
        offset = 0
        limit = 100
        pages_fetched = 0
        
        while True:
            if pages_fetched >= self.MAX_PAGINATION_PAGES:
                logger.warning(f"JolpicaClient: sprint results pagination limit ({self.MAX_PAGINATION_PAGES}) reached for {season}")
                break

            js = self._get(f"{season}/sprint.json", params={"limit": limit, "offset": offset})
            mr = self._extract_mrdata(js)
            races = mr.get("RaceTable", {}).get("Races", []) or []
            pages_fetched += 1
            
            for r in races:
                rnd = str(r.get("round"))
                if rnd not in all_races:
                    all_races[rnd] = r
                else:
                    existing = all_races[rnd].get("SprintResults", []) or []
                    new_results = r.get("SprintResults", []) or []
                    all_races[rnd]["SprintResults"] = existing + new_results
            
            total = int(mr.get("total", 0))
            offset += limit
            if offset >= total or not races:
                break
        
        return list(all_races.values())
