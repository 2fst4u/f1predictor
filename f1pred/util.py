import logging
import json
import re
import sys
import threading
import time
import itertools
from typing import Any, Dict, Optional
from pathlib import Path
from datetime import timedelta

import requests
import requests_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from colorama import init as colorama_init, Fore, Style

colorama_init(autoreset=True)


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def get_logger(name: str, level: str = "WARNING") -> logging.Logger:
    """Get a logger with the specified level. Call configure_logging() first to set global level."""
    return logging.getLogger(name)


def configure_logging(level: str = "WARNING") -> None:
    """Configure root logging level. Should be called once at startup."""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = level_map.get(level.upper(), logging.WARNING)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,  # Override any existing configuration
    )


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
    Includes protection against large responses (DoS).

    Returns:
      - dict/list parsed from JSON when possible
      - raw text string if JSON parsing fails
    Raises:
      - requests.exceptions.HTTPError for non-2xx responses
      - ValueError if response size exceeds limit
    """
    MAX_SIZE = 10 * 1024 * 1024  # 10MB limit

    # Use stream=True to inspect headers/size before full download
    # Note: When using requests-cache, stream=True might still trigger cache logic,
    # but we manually limit the read size to prevent memory exhaustion.
    with session.get(url, params=params, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()

        # Check Content-Length header if present
        cl = resp.headers.get("Content-Length")
        if cl and int(cl) > MAX_SIZE:
             raise ValueError(f"Response too large ({cl} bytes) from {url}")

        # Read content with limit
        content = bytearray()
        for chunk in resp.iter_content(chunk_size=8192):
            content.extend(chunk)
            if len(content) > MAX_SIZE:
                 raise ValueError(f"Response too large (> {MAX_SIZE} bytes) from {url}")

        # Manually decode and parse
        try:
             # Try to use encoding from headers, default to utf-8
             encoding = resp.encoding or "utf-8"
             text = content.decode(encoding)
             return json.loads(text)
        except (ValueError, UnicodeDecodeError):
             # Fallback: return raw text (decoded if possible)
             try:
                 return content.decode(resp.encoding or "utf-8", errors="replace")
             except Exception:
                 return str(content)


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


def sanitize_for_console(text: str) -> str:
    """
    Remove ANSI escape codes and control characters from text to prevent terminal injection
    and log forging/spoofing.
    """
    # 7-bit C1 ANSI sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', str(text))

    # Replace newlines/tabs with space to preserve readability but prevent structure break
    text = re.sub(r'[\r\n\t\v\f]', ' ', text)

    # Remove remaining control characters (C0: \x00-\x1F, DEL: \x7F, C1: \x80-\x9F)
    control_chars = re.compile(r'[\x00-\x1f\x7f-\x9f]')
    return control_chars.sub('', text)


class StatusSpinner:
    """
    Context manager that displays a spinning animation during long-running blocking operations.
    Suppress INFO logs while spinning to prevent visual clutter, but allows WARNING/ERROR.
    """
    def __init__(self, message: str = "Processing...", delay: float = 0.1):
        self.message = sanitize_for_console(message)
        self.delay = delay
        self.spinner = itertools.cycle(["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"])
        self.running = False
        self.thread = None
        self.start_time = 0.0
        self._previous_log_level = logging.INFO
        self.logger = logging.getLogger()  # Root logger
        self.status = "success"

    def update(self, message: str) -> None:
        """Update the spinner message dynamically."""
        self.message = sanitize_for_console(message)

    def set_status(self, status: str) -> None:
        """Set the completion status (e.g. 'success', 'skipped')."""
        self.status = status

    def spin(self):
        while self.running:
            elapsed = time.time() - self.start_time
            timer = f"({elapsed:.1f}s)"
            # Use ANSI escape code \033[K (or \x1b[K) to clear from cursor to end of line
            # this avoids artifacts when the timer string grows or shrinks
            sys.stdout.write(f"\r{Fore.CYAN}{next(self.spinner)}{Style.RESET_ALL} {self.message} {Style.DIM}{timer}{Style.RESET_ALL}\033[K")
            sys.stdout.flush()
            time.sleep(self.delay)

    def __enter__(self):
        # Suppress INFO logs during spinner
        self._previous_log_level = self.logger.getEffectiveLevel()
        # Only raise level if it was lower than WARNING (e.g. INFO or DEBUG)
        if self._previous_log_level < logging.WARNING:
            self.logger.setLevel(logging.WARNING)

        self.start_time = time.time()
        self.running = True
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.running = False
        if self.thread:
            self.thread.join()

        # Restore logging level
        self.logger.setLevel(self._previous_log_level)

        elapsed = time.time() - self.start_time
        time_str = f"({elapsed:.1f}s)"

        # Clear line and print final status.
        # \033[K clears to end of line, ensuring previous spinner text is gone
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

        if exc_type:
            print(f"{Fore.RED}✖{Style.RESET_ALL} {self.message} {Style.DIM}{time_str}{Style.RESET_ALL} (Failed)")
        elif self.status == "skipped":
            print(f"{Fore.YELLOW}⚠{Style.RESET_ALL} {self.message} {Style.DIM}{time_str}{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}✔{Style.RESET_ALL} {self.message} {Style.DIM}{time_str}{Style.RESET_ALL}")
