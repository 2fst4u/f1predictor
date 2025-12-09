from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import time

import numpy as np
import pandas as pd

from .util import get_logger
from .data.jolpica import JolpicaClient
from .data.open_meteo import OpenMeteoClient
from .data.openf1 import OpenF1Client
from .data.fastf1_backend import get_event, get_session_times
from .roster import derive_roster  # single source of truth for roster

logger = get_logger(__name__)

# In-process cache to avoid re-building history multiple times in the same run
# Key: (season, cutoff-date string, sorted roster driver ids)
_HIST_CACHE: Dict[Tuple[int, str, Tuple[str, ...]], pd.DataFrame] = {}

# Cache for historical weather per (season, round) to avoid repeated Open-Meteo calls in the same run
_WEATHER_EVENT_CACHE: Dict[Tuple[int, int], Dict[str, float]] = {}


def exponential_weights(dates: List[datetime], ref_date: datetime, half_life_days: int) -> np.ndarray:
    ages = np.array([(ref_date - d).days if isinstance(d, datetime) else 0 for d in dates], dtype=float)
    return np.power(0.5, ages / max(1, half_life_days))


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "driverId",
            "constructorId",
            "form_index",
            "team_form_index",
            "driver_team_form_index",
            "team_tenure_events",
            "weather_beta_temp",
            "weather_beta_pressure",
            "weather_beta_wind",
            "weather_beta_rain",
            "weather_effect",
            "wet_skill",
            "cold_skill",
            "wind_skill",
            "pressure_skill",
            "teammate_delta",
            "grid_finish_delta",
            "session_type",
            "is_race",
            "is_qualifying",
            "is_sprint",
        ]
    )


# ... remainder of features.py unchanged ...
