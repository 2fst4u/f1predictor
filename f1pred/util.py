import logging
import json
from typing import Any, Dict, Optional
from pathlib import Path
from datetime import timedelta

import requests
import requests_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from colorama import init as colorama_init

colorama_init(autoreset=True)


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def init_caches(cfg, disable_cache: bool = False) -> None:
    """
    Install a global HTTP cache with sensible per-URL TTLs:
      - Forecast endpoints: short TTL (hours)
      - Historical/static data: longer TTL (days)

    Only caches GET requests. Respects stale-if-error settings from config.
    """
    if disable_cache:
        return

    # TTLs from config
    forecast_hours = cfg.caching.requests_cache.expire_after.get("forecast_hours", 6)
    historical_days = cfg.caching.requests_cache.expire_after.get("historical_days", 30)

    default_expire_seconds = int(timedelta(days=historical_days).total_seconds())
    forecast_expire_seconds = int(timedelta(hours=forecast_hours).total_seconds())

    # Build per-URL TTL mapping
    ds = cfg.data_sources
    urls_expire_after: Dict[str, int] = {}

    # Open-Meteo
    if getattr(ds.open_meteo, "forecast_url", None):
        urls_expire_after[ds.open_meteo.forecast_url] = forecast_expire_seconds
    if getattr(ds.open_meteo, "historical_weather_url", None):
        urls_expire_after[ds.open_meteo.historical_weather_url] = default_expire_seconds
    if getattr(ds.open_meteo, "historical_forecast_url", None):
        urls_expire_after[ds.open_meteo.historical_forecast_url] = default_expire_seconds
    if getattr(ds.open_meteo, "elevation_url", None):
        urls_expire_after[ds.open_meteo.elevation_url] = default_expire_seconds
    if getattr(ds.open_meteo, "geocoding_url", None):
        urls_expire_after[ds.open_meteo.geocoding_url] = default_expire_seconds

    # Jolpica
    if getattr(ds.jolpica, "base_url", None):
        urls_expire_after[ds.jolpica.base_url] = default_expire_seconds

    # OpenF1 (historical)
    if getattr(ds.openf1, "base_url", None):
        urls_expire_after[ds.openf1.base_url] = default_expire_seconds

    backend = cfg.caching.requests_cache.backend
    cache_path = Path(cfg.paths.cache_dir) / "http_cache"

    # Install global cache with per-URL TTLs; cache only GET
    requests_cache.install_cache(
        cache_name=str(cache_path),
        backend=backend,
        allowable_methods=("GET",),
        allowable_codes=cfg.caching.requests_cache.allowable_codes,
        stale_if_error=cfg.caching.requests_cache.stale_if_error,
        expire_after=default_expire_seconds,
        urls_expire_after=urls_expire_after,
    )


def session_with_retries(
    total: int = 5,
    connect: int = 3,
    read: int = 3,
    backoff: float = 0.3,
    status_forcelist=(429, 500, 502, 503, 504),
    pool_connections: int = 10,
    pool_maxsize: int = 10,
) -> requests.Session:
    """
    Create a requests.Session with robust retry policy for GET requests.
    - Respects Retry-After headers
    - Separate connect/read retry counts
    - Reasonable pool sizes
    """
    s = requests.Session()
    retries = Retry(
        total=total,
        connect=connect,
        read=read,
        backoff_factor=backoff,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=pool_connections, pool_maxsize=pool_maxsize)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def http_get_json(session: requests.Session, url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Any:
    """
    GET a URL and return parsed JSON; falls back safely if the payload isn't valid JSON.
    Returns:
      - dict/list parsed from JSON when possible
      - raw text string if JSON parsing fails
    Raises:
      - requests.exceptions.HTTPError for non-2xx responses
    """
    resp = session.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    # Try JSON first
    try:
        return resp.json()
    except ValueError:
        # Fallback: attempt manual parse; else return text
        try:
            return json.loads(resp.text)
        except Exception:
            return resp.text


def clear_http_cache() -> None:
    """
    Clear the installed requests-cache backend if present.
    Safe to call even if no cache is installed.
    """
    try:
        requests_cache.clear()
    except Exception:
        pass


def safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default