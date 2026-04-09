from dataclasses import dataclass
from typing import List, Any, Dict, Optional
from pathlib import Path
import os
import yaml


# -----------------------
# Data model
# -----------------------
@dataclass
class AppSettings:
    model_version: str
    random_seed: int
    timezone: str
    live_refresh_seconds: int
    webhook_debounce_seconds: int = 300
    log_level: str = "WARNING"
    auto_refresh_seconds: int = 3600


@dataclass
class Paths:
    cache_dir: str

    @property
    def fastf1_cache(self) -> str:
        """FastF1 cache lives under the main cache directory."""
        return str(Path(self.cache_dir) / "fastf1")


@dataclass
class Jolpica:
    base_url: str
    timeout_seconds: int
    rate_limit_sleep: float
    enabled: bool


@dataclass
class FastF1Cfg:
    enabled: bool


@dataclass
class OpenMeteo:
    forecast_url: str
    historical_weather_url: str
    historical_forecast_url: str
    geocoding_url: str
    enabled: bool
    temperature_unit: str
    windspeed_unit: str
    precipitation_unit: str


@dataclass
class DataSources:
    jolpica: Jolpica
    fastf1: FastF1Cfg
    open_meteo: OpenMeteo


@dataclass
class RequestsCacheCfg:
    backend: str
    expire_after: dict
    allowable_codes: List[int]
    stale_if_error: bool


@dataclass
class PredictionCacheCfg:
    max_entries: int


@dataclass
class Caching:
    requests_cache: RequestsCacheCfg
    prediction_cache: PredictionCacheCfg


@dataclass
class RecencyHalfLives:
    base: int = 120
    team: int = 240
    weather: Optional[int] = 30


@dataclass
class MonteCarlo:
    draws: int


@dataclass
class FeaturesCfg:
    include_fastf1_fill: bool
    include_circuit_elevation: bool
    include_weather_ensemble: bool


@dataclass
class TargetsCfg:
    session_types: List[str]


@dataclass
class EnsembleCfg:
    w_elo: float = 0.2
    w_bt: float = 0.2
    w_mixed: float = 0.2
    w_gbm: float = 0.4
    min_std: float = 0.05


@dataclass
class SimulationCfg:
    noise_factor: float = 0.15
    min_noise: float = 0.05
    max_penalty_base: float = 20.0


@dataclass
class BlendingCfg:
    gbm_weight: float = 0.75
    baseline_weight: float = 0.25
    baseline_team_factor: float = 0.3
    baseline_driver_team_factor: float = 0.2
    grid_factor: float = 0.8
    current_season_weight: float = 8.0
    current_season_qualifying_weight: float = 8.0
    current_quali_factor: float = 0.5
    analytical_win_weight: float = 0.5


@dataclass
class DNFCfg:
    alpha: float = 2.0
    beta: float = 8.0
    driver_weight: float = 0.6
    team_weight: float = 0.4
    clip_min: float = 0.02
    clip_max: float = 0.35


@dataclass
class Modelling:
    recency_half_life_days: RecencyHalfLives
    monte_carlo: MonteCarlo
    features: FeaturesCfg
    targets: TargetsCfg
    ensemble: EnsembleCfg
    simulation: SimulationCfg
    blending: BlendingCfg
    dnf: DNFCfg
    pace_scale: float = 1.0  # Deprecated - kept for backwards compatibility


@dataclass
class Backtesting:
    enabled: bool
    seasons: Any
    metrics: List[str]


@dataclass
class CalibrationCfg:
    enabled: bool
    lookback_window_days: int
    frequency_hours: int
    weights_file: str





@dataclass
class AppConfig:
    app: AppSettings
    paths: Paths
    data_sources: DataSources
    caching: Caching
    modelling: Modelling
    backtesting: Backtesting
    calibration: CalibrationCfg


# -----------------------
# Helpers and validation
# -----------------------
_ALLOWED_SESSIONS = {"qualifying", "race", "sprint_qualifying", "sprint"}
_ALLOWED_TEMP_UNITS = {"celsius", "fahrenheit"}
_ALLOWED_WIND_UNITS = {"kmh", "ms", "mph", "kn"}
_ALLOWED_PRECIP_UNITS = {"mm", "inch"}

def _norm_path(p: str, base_dir: Optional[Path] = None) -> str:
    path = Path(os.path.expandvars(os.path.expanduser(p)))
    if base_dir and not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _require(d: Dict, key: str, ctx: str):
    if key not in d:
        raise KeyError(f"Missing required key '{ctx}.{key}'")
    return d[key]


def _is_http_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))


def load_config(path: str) -> AppConfig:
    config_path = Path(path).resolve()
    base_dir = config_path.parent

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    errors: List[str] = []

    for section in ("app", "paths", "data_sources", "caching", "modelling", "backtesting", "calibration"):
        if section not in cfg:
            errors.append(f"Missing top-level section '{section}'")

    if errors:
        raise ValueError("Invalid config:\n- " + "\n- ".join(errors))

    # Paths
    paths_in = cfg["paths"]
    if "cache_dir" not in paths_in:
        errors.append("Missing paths.cache_dir")
    if not errors:
        # Only normalise cache_dir; fastf1_cache is derived via property
        paths_in = {"cache_dir": _norm_path(paths_in["cache_dir"], base_dir)}

    # Data source URL sanity and unit checks
    ds_in = cfg["data_sources"]
    try:
        jol = _require(ds_in, "jolpica", "data_sources")
        if not _is_http_url(jol.get("base_url", "")):
            errors.append("data_sources.jolpica.base_url must be http(s) URL")

        om = _require(ds_in, "open_meteo", "data_sources")
        for ukey in (
            "forecast_url",
            "historical_weather_url",
            "historical_forecast_url",
            "geocoding_url",
        ):
            if not _is_http_url(om.get(ukey, "")):
                errors.append(f"data_sources.open_meteo.{ukey} must be http(s) URL")
        # Unit validation
        tu = om.get("temperature_unit", "celsius")
        wu = om.get("windspeed_unit", "kmh")
        pu = om.get("precipitation_unit", "mm")
        if tu not in _ALLOWED_TEMP_UNITS:
            errors.append(f"data_sources.open_meteo.temperature_unit must be one of {sorted(_ALLOWED_TEMP_UNITS)}")
        if wu not in _ALLOWED_WIND_UNITS:
            errors.append(f"data_sources.open_meteo.windspeed_unit must be one of {sorted(_ALLOWED_WIND_UNITS)}")
        if pu not in _ALLOWED_PRECIP_UNITS:
            errors.append(f"data_sources.open_meteo.precipitation_unit must be one of {sorted(_ALLOWED_PRECIP_UNITS)}")

    except KeyError as e:
        errors.append(str(e))

    # Modelling targets
    try:
        tgt = _require(cfg["modelling"], "targets", "modelling")
        stypes = tgt.get("session_types", [])
        if not isinstance(stypes, list) or not all(isinstance(x, str) for x in stypes):
            errors.append("modelling.targets.session_types must be a list of strings")
        else:
            unknown = [x for x in stypes if x not in _ALLOWED_SESSIONS]
            if unknown:
                errors.append(
                    f"modelling.targets.session_types contains unknown entries: {unknown}. "
                    f"Allowed: {sorted(_ALLOWED_SESSIONS)}"
                )
    except KeyError as e:
        errors.append(str(e))

    # Caching allowable codes list
    try:
        rc = _require(cfg["caching"], "requests_cache", "caching")
        codes = rc.get("allowable_codes", [])
        if not isinstance(codes, list) or not all(isinstance(x, int) for x in codes):
            errors.append("caching.requests_cache.allowable_codes must be a list of integers")
    except KeyError as e:
        errors.append(str(e))

    # Security Limits: Resource Exhaustion & DoS Protection
    # Enforce safe limits on configuration values that affect resource usage
    try:
        # Monte Carlo draws: Limit memory usage (DoS)
        mc_cfg = _require(cfg["modelling"], "monte_carlo", "modelling")
        if "draws" in mc_cfg:
            draws = mc_cfg["draws"]
            if not isinstance(draws, int) or draws < 100 or draws > 100_000:
                errors.append("modelling.monte_carlo.draws must be between 100 and 100,000 to prevent resource exhaustion")

        # Live Refresh: Limit API call frequency (Abuse Prevention)
        # Using cfg["app"] is safe here because top-level check passed
        app_cfg = cfg["app"]
        if "live_refresh_seconds" in app_cfg:
            refresh = app_cfg["live_refresh_seconds"]
            if not isinstance(refresh, int) or refresh < 10:
                errors.append("app.live_refresh_seconds must be at least 10 seconds to prevent API abuse")

        # Auto Refresh: Background prediction poll interval
        if "auto_refresh_seconds" in app_cfg:
            auto_refresh = app_cfg["auto_refresh_seconds"]
            if not isinstance(auto_refresh, int) or auto_refresh < 60:
                errors.append("app.auto_refresh_seconds must be at least 60 seconds")

        # Webhook Debounce
        if "webhook_debounce_seconds" in app_cfg:
            debounce = app_cfg["webhook_debounce_seconds"]
            if not isinstance(debounce, int) or debounce < 0:
                errors.append("app.webhook_debounce_seconds must be a non-negative integer")

    except KeyError as e:
        errors.append(str(e))

    if errors:
        raise ValueError("Invalid config:\n- " + "\n- ".join(errors))

    # Construct dataclasses
    app = AppSettings(**cfg["app"])
    paths = Paths(**paths_in)
    data_sources = DataSources(
        jolpica=Jolpica(**ds_in["jolpica"]),
        fastf1=FastF1Cfg(**ds_in["fastf1"]),
        open_meteo=OpenMeteo(**ds_in["open_meteo"]),
    )
    rc_dc = RequestsCacheCfg(**cfg["caching"]["requests_cache"])
    pc_dc = PredictionCacheCfg(**cfg["caching"].get("prediction_cache", {"max_entries": 50}))
    caching = Caching(requests_cache=rc_dc, prediction_cache=pc_dc)
    # Recency half-lives: fully defaulted, config can override
    rh_in = cfg["modelling"].get("recency_half_life_days", {})
    rh = RecencyHalfLives(**rh_in) if rh_in else RecencyHalfLives()

    mc = MonteCarlo(**cfg["modelling"]["monte_carlo"])
    feat = FeaturesCfg(**cfg["modelling"]["features"])
    tgt_dc = TargetsCfg(**cfg["modelling"]["targets"])

    # All calibratable sections: dataclass defaults are the source of truth.
    # Config YAML may supply non-calibratable fields only (e.g. min_std, clip_min).
    ens_in = cfg["modelling"].get("ensemble", {})
    ens_dc = EnsembleCfg(**ens_in)

    sim_in = cfg["modelling"].get("simulation", {})
    sim_dc = SimulationCfg(**sim_in)

    blend_in = cfg["modelling"].get("blending", {})
    blend_dc = BlendingCfg(**blend_in)

    dnf_in = cfg["modelling"].get("dnf", {})
    dnf_dc = DNFCfg(**dnf_in)

    pace_scale = float(cfg["modelling"].get("pace_scale", 1.0))  # Deprecated
    modelling = Modelling(
        recency_half_life_days=rh,
        monte_carlo=mc,
        features=feat,
        targets=tgt_dc,
        ensemble=ens_dc,
        simulation=sim_dc,
        blending=blend_dc,
        dnf=dnf_dc,
        pace_scale=pace_scale,
    )
    backtesting = Backtesting(**cfg["backtesting"])

    cal_in = cfg["calibration"]
    if "weights_file" in cal_in:
        cal_in["weights_file"] = _norm_path(cal_in["weights_file"], base_dir)
    calibration = CalibrationCfg(**cal_in)

    return AppConfig(
        app=app,
        paths=paths,
        data_sources=data_sources,
        caching=caching,
        modelling=modelling,
        backtesting=backtesting,
        calibration=calibration,
    )
