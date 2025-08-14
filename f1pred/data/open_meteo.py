from __future__ import annotations
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

import pandas as pd

from ..util import session_with_retries, http_get_json, get_logger, safe_float

logger = get_logger(__name__)


def _valid_lat_lon(lat: float, lon: float) -> bool:
    try:
        return -90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0
    except Exception:
        return False


class OpenMeteoClient:
    def __init__(self,
                 forecast_url: str,
                 historical_weather_url: str,
                 historical_forecast_url: str,
                 elevation_url: str,
                 geocoding_url: str,
                 timeout: int = 30,
                 temperature_unit: str = "celsius",
                 windspeed_unit: str = "kmh",
                 precipitation_unit: str = "mm"):
        self.forecast_url = forecast_url
        self.historical_weather_url = historical_weather_url
        self.historical_forecast_url = historical_forecast_url
        self.elevation_url = elevation_url
        self.geocoding_url = geocoding_url
        self.timeout = timeout
        self.temperature_unit = temperature_unit
        self.windspeed_unit = windspeed_unit
        self.precipitation_unit = precipitation_unit
        self.session = session_with_retries()

    def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        if not _valid_lat_lon(lat, lon):
            logger.info(f"OpenMeteoClient.get_elevation: invalid coordinates lat={lat}, lon={lon}")
            return None
        js = http_get_json(self.session, self.elevation_url,
                           params={"latitude": lat, "longitude": lon},
                           timeout=self.timeout)
        elev = None
        try:
            elev = js.get("elevation", [None])[0] if isinstance(js, dict) else None
        except Exception:
            pass
        return safe_float(elev, None)

    def _fetch_hourly_df(self, base_url: str, params: Dict[str, Any]) -> pd.DataFrame:
        try:
            js = http_get_json(self.session, base_url, params=params, timeout=self.timeout)
            if not isinstance(js, dict):
                logger.info(f"OpenMeteoClient: unexpected response type for {base_url}")
                return pd.DataFrame(columns=["time"])
            hourly = js.get("hourly", {})
            if not isinstance(hourly, dict):
                return pd.DataFrame(columns=["time"])
            time_index = hourly.get("time", [])
            df = pd.DataFrame({"time": pd.to_datetime(time_index)}) if time_index else pd.DataFrame(columns=["time"])
            for k, v in hourly.items():
                if k == "time":
                    continue
                df[k] = v
            return df
        except Exception as e:
            logger.info(f"OpenMeteoClient: failed fetching {base_url}: {e}")
            return pd.DataFrame(columns=["time"])

    def get_forecast(self, lat: float, lon: float, start: datetime, end: datetime, tz: str = "auto") -> pd.DataFrame:
        if not _valid_lat_lon(lat, lon):
            logger.info(f"OpenMeteoClient.get_forecast: invalid coordinates lat={lat}, lon={lon}")
            return pd.DataFrame()
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
                       "pressure_msl", "surface_pressure", "precipitation", "rain", "showers", "snowfall",
                       "weather_code", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "cloud_cover"],
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "timezone": tz,
            "models": "auto",
            "temperature_unit": self.temperature_unit,
            "windspeed_unit": self.windspeed_unit,
            "precipitation_unit": self.precipitation_unit,
            "timeformat": "iso8601",
        }
        return self._fetch_hourly_df(self.forecast_url, params)

    def get_historical_weather(self, lat: float, lon: float, start: datetime, end: datetime, tz: str = "auto") -> pd.DataFrame:
        if not _valid_lat_lon(lat, lon):
            logger.info(f"OpenMeteoClient.get_historical_weather: invalid coordinates lat={lat}, lon={lon}")
            return pd.DataFrame()
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
                       "surface_pressure", "precipitation", "rain", "snowfall", "cloud_cover", "wind_speed_10m",
                       "wind_direction_10m", "wind_gusts_10m"],
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "timezone": tz,
            "temperature_unit": self.temperature_unit,
            "windspeed_unit": self.windspeed_unit,
            "precipitation_unit": self.precipitation_unit,
            "timeformat": "iso8601",
        }
        return self._fetch_hourly_df(self.historical_weather_url, params)

    def get_historical_forecast(self, lat: float, lon: float, start: datetime, end: datetime, tz: str = "auto") -> pd.DataFrame:
        if not _valid_lat_lon(lat, lon):
            logger.info(f"OpenMeteoClient.get_historical_forecast: invalid coordinates lat={lat}, lon={lon}")
            return pd.DataFrame()
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
                       "pressure_msl", "surface_pressure", "precipitation", "rain", "snowfall", "cloud_cover",
                       "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "timezone": tz,
            "models": "auto",
            "temperature_unit": self.temperature_unit,
            "windspeed_unit": self.windspeed_unit,
            "precipitation_unit": self.precipitation_unit,
            "timeformat": "iso8601",
        }
        return self._fetch_hourly_df(self.historical_forecast_url, params)

    @staticmethod
    def aggregate_for_session(weather_df: pd.DataFrame, session_start: datetime, session_end: datetime) -> Dict[str, float]:
        if weather_df is None or weather_df.empty:
            return {}
        if "time" not in weather_df.columns:
            return {}
        mask = (weather_df["time"] >= session_start) & (weather_df["time"] <= session_end)
        subset = weather_df.loc[mask]
        if subset.empty:
            try:
                nearest = weather_df.iloc[(weather_df["time"] - session_start).abs().argsort()[:3]]
                subset = nearest
            except Exception:
                subset = weather_df.head(3)

        def _mean(col: str, default=None):
            return float(subset[col].mean()) if col in subset.columns and not subset[col].isna().all() else (default if default is not None else float("nan"))

        def _min(col: str, default=None):
            return float(subset[col].min()) if col in subset.columns and not subset[col].isna().all() else (default if default is not None else float("nan"))

        def _max(col: str, default=None):
            return float(subset[col].max()) if col in subset.columns and not subset[col].isna().all() else (default if default is not None else float("nan"))

        def _sum(col: str, default=0.0):
            return float(subset[col].sum()) if col in subset.columns and not subset[col].isna().all() else float(default)

        pressure_mean = _mean("surface_pressure")
        if pressure_mean != pressure_mean:
            pressure_mean = _mean("pressure_msl")

        agg = {
            "temp_mean": _mean("temperature_2m"),
            "temp_min": _min("temperature_2m"),
            "temp_max": _max("temperature_2m"),
            "pressure_mean": pressure_mean if pressure_mean == pressure_mean else float("nan"),
            "wind_mean": _mean("wind_speed_10m"),
            "wind_gust_max": _max("wind_gusts_10m"),
            "precip_sum": _sum("precipitation", 0.0),
            "rain_sum": _sum("rain", 0.0),
            "cloud_mean": _mean("cloud_cover"),
            "humidity_mean": _mean("relative_humidity_2m"),
        }
        wet_amount = (agg["rain_sum"] if agg["rain_sum"] == agg["rain_sum"] else 0.0) + (agg["precip_sum"] if agg["precip_sum"] == agg["precip_sum"] else 0.0)
        agg["is_wet_prob"] = 1.0 if wet_amount > 0.5 else 0.0
        return agg