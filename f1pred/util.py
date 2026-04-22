import logging
import json
import os
import re
import sys
import threading
import time
import itertools
import importlib.metadata
import hashlib
import io
from typing import Any, Dict, Optional, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import is_dataclass, asdict

import requests
import requests_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from colorama import init as colorama_init, Fore, Style

colorama_init(autoreset=True)

HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"

try:
    __version__ = importlib.metadata.version("f1predictor")
except importlib.metadata.PackageNotFoundError:
    # Fallback for local development if not installed
    __version__ = "0.0.0.dev0"

USER_AGENT = f"f1predictor/{__version__}"


# --- Logic fingerprint ---------------------------------------------------
# We tie the prediction cache and the broadcast-state file to a short hash
# of the source files whose behaviour, if it changes, MUST invalidate any
# cached prediction. This avoids the class of bug where a roster/feature
# fix ships in a new image but stale predictions from a previous image
# still render in the UI.
#
# Included modules:
#   - f1pred.roster           : determines which drivers are in the grid.
#                                A bug fix here (e.g. reserve-driver leaks)
#                                must invalidate cached predictions that
#                                were built against the buggy roster.
#   - f1pred.features         : feature construction feeding the models.
#
# We also include the running package version so that any deploy, even if
# the above files are unchanged, busts the cache deterministically.
#
# The fingerprint is short (12 hex chars) so it's readable in cache
# filenames / logs, but wide enough to make accidental collisions absurdly
# unlikely.

_LOGIC_FINGERPRINT: Optional[str] = None


def logic_fingerprint() -> str:
    """Return a short, stable hash of the prediction-logic source + version.

    Computed once per process and memoised. Used as a cache-key component
    so any code change in roster/feature modules, or any version bump,
    automatically invalidates cached predictions.
    """
    global _LOGIC_FINGERPRINT
    if _LOGIC_FINGERPRINT is not None:
        return _LOGIC_FINGERPRINT

    parts: list[str] = [f"version={__version__}"]

    # Resolve module source paths relative to this file so import-order
    # quirks don't matter.
    here = Path(__file__).resolve().parent
    for rel in ("roster.py", "features.py"):
        src = here / rel
        try:
            h = hashlib.sha256(src.read_bytes()).hexdigest()
            parts.append(f"{rel}:{h}")
        except Exception:
            # If a module file is missing, fall back to its name only.
            # The version component still busts caches across deploys.
            parts.append(f"{rel}:missing")

    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    _LOGIC_FINGERPRINT = digest[:12]
    return _LOGIC_FINGERPRINT

_UMASK_LOCK = threading.Lock()


def ensure_dirs(*paths: str) -> None:
    logger = logging.getLogger(__name__)
    # Set umask to 0o077 to ensure created directories have 0o700 permissions atomically.
    # This prevents a race condition where directories might briefly exist with wider permissions.
    # Use a lock to ensure thread safety for the global process umask.
    with _UMASK_LOCK:
        old_umask = os.umask(0o077)
        try:
            for p in paths:
                path_obj = Path(p)
                path_obj.mkdir(parents=True, exist_ok=True)
                # Explicit chmod as a backup in case mkdir ignores umask (rare but possible)
                # or if the directory already existed with wrong permissions.
                try:
                    path_obj.chmod(0o700)
                except Exception as e:
                    logger.warning(f"Failed to set secure permissions (0o700) on {p}: {e}. Cache may be insecure.")
        finally:
            os.umask(old_umask)


def sanitize_for_console(text: str) -> str:
    """
    Remove ANSI escape codes and control characters from text to prevent terminal injection
    and log forging/spoofing.

    Also truncates excessively long inputs to prevent Log Flooding and Terminal DoS.
    """
    s_text = str(text)

    # Truncate to prevent Log Flooding / Resource Exhaustion
    MAX_LOG_LENGTH = 1024
    if len(s_text) > MAX_LOG_LENGTH:
        s_text = s_text[:MAX_LOG_LENGTH] + "...[truncated]"

    # 7-bit C1 ANSI sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', s_text)

    # Replace newlines/tabs with space to preserve readability but prevent structure break
    text = re.sub(r'[\r\n\t\v\f]', ' ', text)

    # Remove remaining control characters (C0: \x00-\x1F, DEL: \x7F, C1: \x80-\x9F)
    control_chars = re.compile(r'[\x00-\x1f\x7f-\x9f]')
    return control_chars.sub('', text)


def get_logger(name: str, level: str = "WARNING") -> logging.Logger:
    """Get a logger with the specified level. Call configure_logging() first to set global level."""
    return logging.getLogger(name)


class SafeLogFormatter(logging.Formatter):
    """
    Log formatter that sanitizes messages to prevent Log Injection.
    Replaces newlines and control characters with spaces.
    """
    def format(self, record: logging.LogRecord) -> str:
        # Format the message using the standard formatter first (interpolates args)
        original_msg = super().format(record)
        # Sanitize the result
        return sanitize_for_console(original_msg)


def configure_logging(level: str = "WARNING") -> None:
    """Configure root logging level. Should be called once at startup."""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = level_map.get(level.upper(), logging.WARNING)

    # Use SafeLogFormatter to prevent log injection
    handler = logging.StreamHandler()
    formatter = SafeLogFormatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)

    logging.basicConfig(
        level=log_level,
        handlers=[handler],
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
    if getattr(ds.open_meteo, "geocoding_url", None):
        urls_expire_after[ds.open_meteo.geocoding_url] = default_expire_seconds

    # Jolpica
    if getattr(ds.jolpica, "base_url", None):
        base_url = ds.jolpica.base_url.rstrip("/")
        urls_expire_after[base_url] = default_expire_seconds
        # Shorten TTL for dynamic "next" and "last" pointers
        urls_expire_after[f"{base_url}/current/next.json"] = forecast_expire_seconds
        urls_expire_after[f"{base_url}/current/last.json"] = forecast_expire_seconds

    backend = cfg.caching.requests_cache.backend
    cache_path = Path(cfg.paths.cache_dir) / "http_cache"

    # Install global cache with per-URL TTLs; cache only GET
    # Use JSON serializer to prevent insecure deserialization (pickle) vulnerabilities
    requests_cache.install_cache(
        cache_name=str(cache_path),
        backend=backend,
        serializer="json",
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
    s.headers["User-Agent"] = USER_AGENT
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


def http_get_json(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    include_metadata: bool = False,
    force_refresh: bool = False,
) -> Any:
    """
    GET a URL and return parsed JSON; falls back safely if the payload isn't valid JSON.
    Includes protection against large responses (DoS).

    Args:
        force_refresh: If True, bypass the HTTP cache and fetch a fresh response from the
            origin server (equivalent to requests_cache's force_refresh parameter). Use this
            when a previously cached empty/stale response needs to be re-validated.

    Returns:
      - If include_metadata=False (default): dict/list parsed from JSON when possible,
        else raw text string.
      - If include_metadata=True: tuple (data, from_cache).
    Raises:
      - requests.exceptions.HTTPError for non-2xx responses
      - ValueError if response size exceeds limit
    """
    MAX_SIZE = 10 * 1024 * 1024  # 10MB limit

    # Use stream=True to inspect headers/size before full download
    # Note: When using requests-cache, stream=True might still trigger cache logic,
    # but we manually limit the read size to prevent memory exhaustion.
    # force_refresh=True is passed to requests-cache to bypass a stale cached response.
    # Only pass it when the session actually supports it (i.e. is a CachedSession);
    # plain requests.Session raises TypeError on unknown kwargs.
    extra_kwargs: Dict[str, Any] = {}
    if force_refresh and hasattr(session, "cache"):
        extra_kwargs["force_refresh"] = True
    with session.get(url, params=params, timeout=timeout, stream=True, **extra_kwargs) as resp:
        resp.raise_for_status()

        # Check Content-Length header if present
        cl = resp.headers.get("Content-Length")
        if cl:
            try:
                cl_int = int(cl)
            except ValueError:
                # Malformed Content-Length; proceed (we enforce limit during read anyway)
                cl_int = 0

            if cl_int > MAX_SIZE:
                raise ValueError(f"Response too large ({cl} bytes) from {url}")

        # Read content with limit
        content = bytearray()
        for chunk in resp.iter_content(chunk_size=8192):
            content.extend(chunk)
            if len(content) > MAX_SIZE:
                 raise ValueError(f"Response too large (> {MAX_SIZE} bytes) from {url}")

        # Manually decode and parse
        from_cache = getattr(resp, "from_cache", False)
        try:
             # Try to use encoding from headers, default to utf-8
             encoding = resp.encoding or "utf-8"
             text = content.decode(encoding)
             data = json.loads(text)
        except (ValueError, UnicodeDecodeError):
             # Fallback: return raw text (decoded if possible)
             try:
                 data = content.decode(resp.encoding or "utf-8", errors="replace")
             except Exception:
                 data = str(content)

        if include_metadata:
            return data, from_cache
        return data


def safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


class StatusSpinner:
    """
    Context manager that displays a spinning animation during long-running blocking operations.
    Suppress INFO logs while spinning to prevent visual clutter, but allows WARNING/ERROR.
    """
    def __init__(self, message: str = "Processing...", delay: float = 0.1, on_update: Optional[Callable[[str], None]] = None):
        self.message = sanitize_for_console(message)
        self.delay = delay
        self.on_update = on_update
        self.spinner = itertools.cycle(["-", "\\", "|", "/"])
        self.running = False
        self.thread = None
        self.start_time = 0.0
        self._previous_log_level = logging.INFO
        self.logger = logging.getLogger()  # Root logger
        self.status = "success"
        self.is_tty = sys.stdout.isatty()

    def update(self, message: str) -> None:
        """Update the spinner message dynamically."""
        self.message = sanitize_for_console(message)
        if self.on_update:
            self.on_update(self.message)

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

        # Only start animation thread and hide cursor if interactive (TTY)
        if self.is_tty:
            sys.stdout.write(HIDE_CURSOR)
            sys.stdout.flush()
            self.thread = threading.Thread(target=self.spin)
            self.thread.start()

        if self.on_update:
            self.on_update(self.message)

        return self

    def __exit__(self, exc_type, *_):
        self.running = False
        if self.thread:
            self.thread.join()

        # Restore logging level
        self.logger.setLevel(self._previous_log_level)

        elapsed = time.time() - self.start_time
        time_str = f"({elapsed:.1f}s)"

        if self.is_tty:
            # Restore cursor and clear spinner line
            sys.stdout.write(SHOW_CURSOR)
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

        if exc_type:
            print(f"{Fore.RED}x{Style.RESET_ALL} {self.message} {Style.DIM}{time_str}{Style.RESET_ALL} (Failed)")
        elif self.status == "skipped":
            print(f"{Fore.YELLOW}!{Style.RESET_ALL} {self.message} {Style.DIM}{time_str}{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}+{Style.RESET_ALL} {self.message} {Style.DIM}{time_str}{Style.RESET_ALL}")

def print_countdown(seconds: int, message: str = "Refreshing in") -> None:
    """
    Display a countdown timer on the same line.
    Falls back to simple sleep if not a TTY.
    """
    if not sys.stdout.isatty():
        time.sleep(seconds)
        return

    # Hide cursor
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()

    try:
        for i in range(seconds, 0, -1):
            sys.stdout.write(f"\r{Style.DIM}↻ {message} {i}s...{Style.RESET_ALL}\033[K")
            sys.stdout.flush()
            time.sleep(1)

        # Clear line on finish
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
    finally:
        # Restore cursor
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()


class PredictionCache:
    """
    Persistent, disk-based rolling cache for prediction results.
    Uses SHA-256 hashes of input variables as keys.
    """
    def __init__(self, cache_dir: str, max_entries: int = 50):
        self.cache_dir = Path(cache_dir) / "predictions"
        self.max_entries = max_entries
        self.lock = threading.Lock()
        ensure_dirs(str(self.cache_dir))

    def _serialize(self, obj: Any, sort_dict: bool = False) -> Any:
        """Recursively serialize objects for caching."""
        import pandas as pd
        import numpy as np
        if isinstance(obj, pd.DataFrame):
            return obj.to_json(orient="split")
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        if is_dataclass(obj):
            return self._serialize(asdict(obj), sort_dict=sort_dict)
        if isinstance(obj, dict):
            items = sorted(obj.items()) if sort_dict else obj.items()
            return {k: self._serialize(v, sort_dict=sort_dict) for k, v in items}
        if isinstance(obj, list):
            return [self._serialize(i, sort_dict=sort_dict) for i in obj]
        return str(obj) if sort_dict else obj

    def _generate_key(self, inputs: Dict[str, Any]) -> str:
        """Generate a stable SHA-256 hash from a dictionary of inputs."""
        serialized_input = json.dumps(self._serialize(inputs, sort_dict=True), sort_keys=True)
        return hashlib.sha256(serialized_input.encode("utf-8")).hexdigest()


    def get_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached prediction results using an explicit string key."""
        import pandas as pd
        import numpy as np

        # We still hash the explicit key to ensure it's a valid, safe filename
        safe_key = hashlib.sha256(key.encode("utf-8")).hexdigest()
        cache_file = self.cache_dir / f"{safe_key}.json"

        with self.lock:
            if cache_file.exists():
                try:
                    with open(cache_file, "r") as f:
                        data = json.load(f)

                    # Deserialization of DataFrames
                    if "ranked" in data and isinstance(data["ranked"], str):
                        data["ranked"] = pd.read_json(io.StringIO(data["ranked"]), orient="split")
                    if "prob_matrix" in data and isinstance(data["prob_matrix"], list):
                        data["prob_matrix"] = np.array(data["prob_matrix"])
                    if "pairwise" in data and isinstance(data["pairwise"], list):
                        data["pairwise"] = np.array(data["pairwise"])

                    # Update modification time to keep it in the rolling cache
                    cache_file.touch()
                    return data
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Failed to load prediction cache {safe_key}: {e}")
        return None

    def set_by_key(self, key: str, results: Dict[str, Any]) -> None:
        """Save prediction results to the cache using an explicit string key."""
        safe_key = hashlib.sha256(key.encode("utf-8")).hexdigest()
        cache_file = self.cache_dir / f"{safe_key}.json"

        with self.lock:
            try:
                # Rolling deletion if limit exceeded
                files = sorted(self.cache_dir.glob("*.json"), key=os.path.getmtime)
                if len(files) >= self.max_entries:
                    # Delete the oldest one(s)
                    for f in files[:len(files) - self.max_entries + 1]:
                        try:
                            f.unlink()
                        except Exception:
                            pass

                # Save new entry
                serialized_data = self._serialize(results)
                with open(cache_file, "w") as f:
                    json.dump(serialized_data, f)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to save prediction cache {safe_key}: {e}")

    def get(self, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve cached prediction results if they exist."""
        import pandas as pd
        import numpy as np
        key = self._generate_key(inputs)
        cache_file = self.cache_dir / f"{key}.json"

        with self.lock:
            if cache_file.exists():
                try:
                    with open(cache_file, "r") as f:
                        data = json.load(f)

                    # Deserialization of DataFrames
                    if "ranked" in data and isinstance(data["ranked"], str):
                        data["ranked"] = pd.read_json(io.StringIO(data["ranked"]), orient="split")
                    if "prob_matrix" in data and isinstance(data["prob_matrix"], list):
                        data["prob_matrix"] = np.array(data["prob_matrix"])
                    if "pairwise" in data and isinstance(data["pairwise"], list):
                        data["pairwise"] = np.array(data["pairwise"])

                    # Update modification time to keep it in the rolling cache
                    cache_file.touch()
                    return data
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Failed to load prediction cache {key}: {e}")
        return None

    def set(self, inputs: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Save prediction results to the cache."""
        key = self._generate_key(inputs)
        cache_file = self.cache_dir / f"{key}.json"

        with self.lock:
            try:
                # Rolling deletion if limit exceeded
                files = sorted(self.cache_dir.glob("*.json"), key=os.path.getmtime)
                if len(files) >= self.max_entries:
                    # Delete the oldest one(s)
                    for f in files[:len(files) - self.max_entries + 1]:
                        try:
                            f.unlink()
                        except Exception:
                            pass

                # Save new entry
                serialized_data = self._serialize(results)
                with open(cache_file, "w") as f:
                    json.dump(serialized_data, f)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to save prediction cache {key}: {e}")
