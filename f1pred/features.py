from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import numpy as np
import pandas as pd

from .util import get_logger
from .data.jolpica import JolpicaClient
from .data.open_meteo import OpenMeteoClient
from .data.openf1 import OpenF1Client
from .data.fastf1_backend import get_event, get_session_times
from .roster import derive_roster  # single source of truth

logger = get_logger(__name__)

# ... existing functions ...

def build_roster(jc: JolpicaClient, season: str, rnd: str) -> pd.DataFrame:
    """
    Thin wrapper around derive_roster to keep Jolpica client as data-access only.
    All derivation logic lives in f1pred/roster.py.
    """
    entries = derive_roster(jc, season, rnd, prefer_same_round_qualifying=True)
    df = pd.DataFrame(entries)
    if not df.empty:
        # create human-readable name
        def _mk(n, f): return f"{n or ''} {f or ''}".strip()
        df["name"] = df.apply(lambda x: _mk(x.get("givenName"), x.get("familyName")), axis=1)
    return df