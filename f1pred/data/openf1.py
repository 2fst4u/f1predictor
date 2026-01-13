from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd

import requests_cache
from ..util import session_with_retries, http_get_json, get_logger

logger = get_logger(__name__)


class OpenF1Client:
    def __init__(self, base_url: str, timeout: int = 30, enabled: bool = True):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.enabled = enabled
        self.session = session_with_retries()

    def _validate_season(self, season: Any) -> int:
        """Ensure season is a valid year."""
        try:
            s = int(str(season).strip())
            # Basic range check - F1 started in 1950, and we don't expect season 3000 yet
            if s < 1950 or s > 2100:
                raise ValueError
            return s
        except (ValueError, TypeError):
            raise ValueError(f"Invalid season: {season}")

    def _validate_round(self, rnd: Any) -> int:
        """Ensure round is a valid positive integer."""
        try:
            r = int(str(rnd).strip())
            if r < 1:
                raise ValueError
            return r
        except (ValueError, TypeError):
            raise ValueError(f"Invalid round: {rnd}")

    def _get(self, endpoint: str, params: Dict[str, Any], use_cache: bool = True) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            if use_cache:
                js = http_get_json(self.session, url, params=params, timeout=self.timeout)
            else:
                with requests_cache.disabled():
                    js = http_get_json(self.session, url, params=params, timeout=self.timeout)
        except Exception as e:
            logger.info(f"OpenF1 GET {endpoint} failed: {e}")
            return []
        if isinstance(js, list):
            return js
        elif isinstance(js, dict) and "data" in js:
            return js["data"]
        else:
            return []

    def _get_meeting_key(self, season: int, round_no: int, use_cache: bool = True) -> Optional[int]:
        """Get meeting_key for a given season and round.
        
        OpenF1 doesn't have a 'round' parameter - we need to fetch all meetings
        for the year and match by index (meetings are returned in date order).
        """
        season_val = self._validate_season(season)
        round_val = self._validate_round(round_no)

        meetings = self._get("meetings", {"year": season_val}, use_cache=use_cache)
        if not meetings:
            return None
        
        # Sort by date to ensure correct order
        try:
            meetings = sorted(meetings, key=lambda m: m.get("date_start", ""))
        except Exception:
            pass
        
        # Round is 1-indexed, list is 0-indexed
        if 1 <= round_val <= len(meetings):
            return meetings[round_val - 1].get("meeting_key")
        return None

    def find_session(self, season: int, round_no: int, session_name: str, use_cache: bool = True) -> Optional[int]:
        """Find session_key for a given season, round, and session name.
        
        This uses the correct OpenF1 API flow:
        1. Get meeting_key from /meetings?year=YYYY
        2. Get sessions from /sessions?meeting_key=XXX
        3. Match by session_name
        """
        if not self.enabled:
            return None
        
        try:
            # Step 1: Get meeting_key for this round
            # Note: _get_meeting_key handles validation of season and round_no
            meeting_key = self._get_meeting_key(season, round_no, use_cache=use_cache)
            if not meeting_key:
                return None
            
            # Step 2: Get sessions for this meeting
            sessions = self._get("sessions", {"meeting_key": meeting_key}, use_cache=use_cache)
        except Exception:
            return None
        
        if not sessions:
            return None
        
        normalized = (session_name or "").strip().lower()
        
        # Direct match
        for s in sessions:
            name = (s.get("session_name") or "").lower()
            if name == normalized:
                return s.get("session_key")
        
        # Heuristics for Sprint Shootout naming
        if "sprint" in normalized and ("qual" in normalized or "shootout" in normalized):
            for s in sessions:
                name = (s.get("session_name") or "").lower()
                if "shootout" in name or "sprint qualifying" in name:
                    return s.get("session_key")
        
        return None

    def get_stints(self, session_key: int) -> pd.DataFrame:
        stints = self._get("stints", {"session_key": session_key})
        return pd.DataFrame(stints)

    def get_laps(self, session_key: int) -> pd.DataFrame:
        laps = self._get("laps", {"session_key": session_key})
        return pd.DataFrame(laps)

    def get_positions(self, session_key: int) -> pd.DataFrame:
        pos = self._get("position", {"session_key": session_key})
        return pd.DataFrame(pos)

    def get_weather(self, session_key: int) -> pd.DataFrame:
        w = self._get("weather", {"session_key": session_key})
        return pd.DataFrame(w)

    def get_drivers(self, session_key: int, use_cache: bool = True) -> pd.DataFrame:
        d = self._get("drivers", {"session_key": session_key}, use_cache=use_cache)
        return pd.DataFrame(d)