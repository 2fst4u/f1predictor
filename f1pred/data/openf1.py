from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd

from ..util import session_with_retries, http_get_json, get_logger

logger = get_logger(__name__)


class OpenF1Client:
    def __init__(self, base_url: str, timeout: int = 30, enabled: bool = True):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.enabled = enabled
        self.session = session_with_retries()

    def _get(self, endpoint: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
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

    def find_session(self, season: int, round_no: int, session_name: str) -> Optional[int]:
        if not self.enabled:
            return None
        try:
            sessions = self._get("sessions", {"year": season, "round": round_no})
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
                if "shootout" in name or "qualifying" in name:
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