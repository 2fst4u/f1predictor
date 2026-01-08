from dataclasses import dataclass
from typing import List, Any, Dict
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


@dataclass
class Paths:
    cache_dir: str
    fastf1_cache: str


@dataclass
class Jolpica:
    base_url: str
    timeout_seconds: int
    rate_limit_sleep: float
    enabled: bool


@dataclass
class OpenF1:
    base_url: str
    timeout_seconds: int
    enabled: bool


@dataclass
class FastF1Cfg:
    enabled: bool


@dataclass
class OpenMeteo:
    forecast_url: str
    historical_weather_url: str
    historical_forecast_url: str
    elevation_url: str
    geocoding_url: str
    enabled: bool
    temperature_unit: str
    windspeed_unit: str
    precipitation_unit: str


@dataclass
class DataSources:
    jolpica: Jolpica
    openf1: OpenF1
    fastf1: FastF1Cfg
    open_meteo: OpenMeteo


@dataclass
class RequestsCacheCfg:
    backend: str
    expire_after: dict
    allowable_codes: List[int]
    stale_if_error: bool


@dataclass
class Caching:
    requests_cache: RequestsCacheCfg


@dataclass
class RecencyHalfLives:
    base: int
    weather: int
    team: int


@dataclass
class MonteCarlo:
    draws: int


@dataclass
class FeaturesCfg:
    include_openf1_tyres: bool
    include_openf1_laps: bool
    include_fastf1_fill: bool
    include_circuit_elevation: bool
    include_weather_ensemble: bool


@dataclass
class TargetsCfg:
    session_types: List[str]


@dataclass
class EnsembleCfg:
    w_elo: float
    w_bt: float
    w_mixed: float
    w_gbm: float
    min_std: float


@dataclass
class SimulationCfg:
    noise_factor: float
    min_noise: float
    max_penalty_base: float


@dataclass
class BlendingCfg:
    gbm_weight: float
    baseline_weight: float
    baseline_team_factor: float
    baseline_driver_team_factor: float


@dataclass
class DNFCfg:
    alpha: float
    beta: float
    driver_weight: float
    team_weight: float
    clip_min: float
    clip_max: float


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
class AppConfig:
    app: AppSettings
    paths: Paths
    data_sources: DataSources
    caching: Caching
    modelling: Modelling
    backtesting: Backtesting


# -----------------------
# Helpers and validation
# -----------------------
_ALLOWED_SESSIONS = {"qualifying", "race", "sprint_qualifying", "sprint"}
_ALLOWED_TEMP_UNITS = {"celsius", "fahrenheit"}
_ALLOWED_WIND_UNITS = {"kmh", "ms", "mph", "kn"}
_ALLOWED_PRECIP_UNITS = {"mm", "inch"}

def _norm_path(p: str) -> str:
    return str(Path(os.path.expandvars(os.path.expanduser(p))))


def _require(d: Dict, key: str, ctx: str):
    if key not in d:
        raise KeyError(f"Missing required key '{ctx}.{key}'")
    return d[key]


def _is_http_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    errors: List[str] = []

    for section in ("app", "paths", "data_sources", "caching", "modelling", "backtesting"):
        if section not in cfg:
            errors.append(f"Missing top-level section '{section}'")

    if errors:
        raise ValueError("Invalid config:\n- " + "\n- ".join(errors))

    # Paths
    paths_in = cfg["paths"]
    for k in ("cache_dir", "fastf1_cache"):
        if k not in paths_in:
            errors.append(f"Missing paths.{k}")
    if not errors:
        paths_in = {k: _norm_path(v) for k, v in paths_in.items()}

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
            "elevation_url",
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

        of1 = _require(ds_in, "openf1", "data_sources")
        if not _is_http_url(of1.get("base_url", "")):
            errors.append("data_sources.openf1.base_url must be http(s) URL")
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

    if errors:
        raise ValueError("Invalid config:\n- " + "\n- ".join(errors))

    # Construct dataclasses
    app = AppSettings(**cfg["app"])
    paths = Paths(**paths_in)
    data_sources = DataSources(
        jolpica=Jolpica(**ds_in["jolpica"]),
        openf1=OpenF1(**ds_in["openf1"]),
        fastf1=FastF1Cfg(**ds_in["fastf1"]),
        open_meteo=OpenMeteo(**ds_in["open_meteo"]),
    )
    rc_dc = RequestsCacheCfg(**cfg["caching"]["requests_cache"])
    caching = Caching(requests_cache=rc_dc)
    rh = RecencyHalfLives(**cfg["modelling"]["recency_half_life_days"])
    mc = MonteCarlo(**cfg["modelling"]["monte_carlo"])
    feat = FeaturesCfg(**cfg["modelling"]["features"])
    tgt_dc = TargetsCfg(**cfg["modelling"]["targets"])
    ens_dc = EnsembleCfg(**cfg["modelling"].get("ensemble", {}))
    
    sim_dc = SimulationCfg(**cfg["modelling"].get("simulation", {}))
    blend_dc = BlendingCfg(**cfg["modelling"].get("blending", {}))
    dnf_dc = DNFCfg(**cfg["modelling"].get("dnf", {}))
    
    pace_scale = float(cfg["modelling"].get("pace_scale", 1.0))  # Default 1.0 (no scaling)
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

    return AppConfig(
        app=app,
        paths=paths,
        data_sources=data_sources,
        caching=caching,
        modelling=modelling,
        backtesting=backtesting,
    )
