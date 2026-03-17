import pandas as pd
from unittest.mock import patch
from datetime import datetime
from f1pred.data.open_meteo import OpenMeteoClient

def test_aggregate_for_session_empty_df():
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    df = pd.DataFrame()
    res = client.aggregate_for_session(df, datetime(2023, 1, 1), datetime(2023, 1, 2))
    assert res == {}

def test_aggregate_for_session_valid_data():
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    df = pd.DataFrame({
        "time": [datetime(2023, 1, 1, 12, 0), datetime(2023, 1, 1, 13, 0), datetime(2023, 1, 1, 14, 0)],
        "temperature_2m": [20.0, 22.0, 24.0],
        "surface_pressure": [1010.0, 1012.0, 1014.0],
        "wind_speed_10m": [10.0, 15.0, 20.0],
        "wind_gusts_10m": [15.0, 20.0, 25.0],
        "precipitation": [0.0, 1.0, 0.0],
        "rain": [0.0, 1.0, 0.0],
        "cloud_cover": [50.0, 60.0, 70.0],
        "relative_humidity_2m": [40.0, 45.0, 50.0]
    })

    start = datetime(2023, 1, 1, 12, 0)
    end = datetime(2023, 1, 1, 13, 0)
    res = client.aggregate_for_session(df, start, end)

    assert res["temp_mean"] == 21.0
    assert res["temp_min"] == 20.0
    assert res["temp_max"] == 22.0
    assert res["pressure_mean"] == 1011.0
    assert res["wind_mean"] == 12.5
    assert res["wind_gust_max"] == 20.0
    assert res["precip_sum"] == 1.0
    assert res["rain_sum"] == 1.0
    assert res["cloud_mean"] == 55.0
    assert res["humidity_mean"] == 42.5


def test_aggregate_for_session_no_time_col():
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    df = pd.DataFrame({"temperature_2m": [20.0, 22.0]})
    res = client.aggregate_for_session(df, datetime(2023, 1, 1), datetime(2023, 1, 2))
    assert res == {}

def test_aggregate_for_session_out_of_bounds():
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    df = pd.DataFrame({
        "time": [datetime(2023, 1, 1, 12, 0)],
        "temperature_2m": [20.0]
    })
    res = client.aggregate_for_session(df, datetime(2023, 1, 2), datetime(2023, 1, 3))
    assert res == {}

def test_aggregate_for_session_fallback_pressure():
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    df = pd.DataFrame({
        "time": [datetime(2023, 1, 1, 12, 0)],
        "temperature_2m": [20.0],
        "pressure_msl": [1010.0]
    })
    res = client.aggregate_for_session(df, datetime(2023, 1, 1, 12, 0), datetime(2023, 1, 1, 12, 0))
    assert res["pressure_mean"] == 1010.0

def test_aggregate_for_session_all_nan():
    import numpy as np
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    df = pd.DataFrame({
        "time": [datetime(2023, 1, 1, 12, 0)],
        "temperature_2m": [np.nan],
        "precipitation": [np.nan]
    })
    res = client.aggregate_for_session(df, datetime(2023, 1, 1, 12, 0), datetime(2023, 1, 1, 12, 0))
    import math
    assert math.isnan(res["temp_mean"])
    assert res["precip_sum"] == 0.0

def test_valid_timezone():
    from f1pred.data.open_meteo import OpenMeteoClient

    # Valid ones
    assert OpenMeteoClient._validate_timezone("UTC") == "UTC"
    assert OpenMeteoClient._validate_timezone("Europe/London") == "Europe/London"
    assert OpenMeteoClient._validate_timezone("America/New_York") == "America/New_York"

    # Invalid character
    assert OpenMeteoClient._validate_timezone("Europe/London;") == "UTC"

    # Too long
    assert OpenMeteoClient._validate_timezone("A" * 65) == "UTC"

    # Empty
    assert OpenMeteoClient._validate_timezone("") == "UTC"
    assert OpenMeteoClient._validate_timezone(None) == "UTC"

def test_valid_lat_lon():
    from f1pred.data.open_meteo import _valid_lat_lon

    # Valid
    assert _valid_lat_lon(0.0, 0.0)
    assert _valid_lat_lon(90.0, 180.0)
    assert _valid_lat_lon(-90.0, -180.0)

    # Invalid lat
    assert not _valid_lat_lon(90.1, 0.0)
    assert not _valid_lat_lon(-90.1, 0.0)

    # Invalid lon
    assert not _valid_lat_lon(0.0, 180.1)
    assert not _valid_lat_lon(0.0, -180.1)

    # Invalid type
    assert not _valid_lat_lon("invalid", 0.0)

def test_normalize_params():
    from f1pred.data.open_meteo import OpenMeteoClient
    params = {
        "latitude": 50.0,
        "hourly": ["temp", "rain"],
        "timezone": "UTC"
    }
    norm = OpenMeteoClient._normalize_params(params)
    assert norm["latitude"] == 50.0
    assert norm["hourly"] == "temp,rain"
    assert norm["timezone"] == "UTC"



@patch("f1pred.data.open_meteo.http_get_json")
def test_fetch_hourly_df_valid(mock_get):
    mock_get.return_value = {
        "hourly": {
            "time": ["2023-01-01T12:00:00Z", "2023-01-01T13:00:00Z"],
            "temperature_2m": [20.0, 22.0]
        }
    }
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    df = client._fetch_hourly_df("test_url", {"hourly": ["temp"]})
    assert len(df) == 2
    assert "time" in df.columns
    assert "temperature_2m" in df.columns

@patch("f1pred.data.open_meteo.http_get_json")
def test_fetch_hourly_df_invalid_response(mock_get):
    mock_get.return_value = ["invalid"]
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    df = client._fetch_hourly_df("test_url", {"hourly": ["temp"]})
    assert len(df) == 0
    assert list(df.columns) == ["time"]

@patch("f1pred.data.open_meteo.http_get_json")
def test_fetch_hourly_df_no_hourly(mock_get):
    mock_get.return_value = {}
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    df = client._fetch_hourly_df("test_url", {"hourly": ["temp"]})
    assert len(df) == 0

@patch("f1pred.data.open_meteo.http_get_json")
def test_fetch_hourly_df_hourly_not_dict(mock_get):
    mock_get.return_value = {"hourly": ["invalid"]}
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    df = client._fetch_hourly_df("test_url", {"hourly": ["temp"]})
    assert len(df) == 0

@patch("f1pred.data.open_meteo.http_get_json")
def test_fetch_hourly_df_exception(mock_get):
    mock_get.side_effect = Exception("Test Error")
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    df = client._fetch_hourly_df("test_url", {"hourly": ["temp"]})
    assert len(df) == 0


@patch.object(OpenMeteoClient, '_fetch_hourly_df')
def test_get_forecast_invalid_coords(mock_fetch):
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    df = client.get_forecast(91.0, 0.0, datetime(2023, 1, 1), datetime(2023, 1, 2))
    assert len(df) == 0
    mock_fetch.assert_not_called()

@patch.object(OpenMeteoClient, '_fetch_hourly_df')
def test_get_forecast_valid(mock_fetch):
    mock_fetch.return_value = pd.DataFrame({"time": []})
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    _ = client.get_forecast(50.0, 0.0, datetime(2023, 1, 1), datetime(2023, 1, 2))
    assert mock_fetch.called

@patch.object(OpenMeteoClient, '_fetch_hourly_df')
def test_get_historical_weather_invalid_coords(mock_fetch):
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    df = client.get_historical_weather(91.0, 0.0, datetime(2023, 1, 1), datetime(2023, 1, 2))
    assert len(df) == 0
    mock_fetch.assert_not_called()

@patch.object(OpenMeteoClient, '_fetch_hourly_df')
def test_get_historical_weather_valid(mock_fetch):
    mock_fetch.return_value = pd.DataFrame({"time": []})
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    _ = client.get_historical_weather(50.0, 0.0, datetime(2023, 1, 1), datetime(2023, 1, 2))
    assert mock_fetch.called

@patch.object(OpenMeteoClient, '_fetch_hourly_df')
def test_get_historical_forecast_invalid_coords(mock_fetch):
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    df = client.get_historical_forecast(91.0, 0.0, datetime(2023, 1, 1), datetime(2023, 1, 2))
    assert len(df) == 0
    mock_fetch.assert_not_called()

@patch.object(OpenMeteoClient, '_fetch_hourly_df')
def test_get_historical_forecast_valid(mock_fetch):
    mock_fetch.return_value = pd.DataFrame({"time": []})
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    _ = client.get_historical_forecast(50.0, 0.0, datetime(2023, 1, 1), datetime(2023, 1, 2))
    assert mock_fetch.called

def test_compute_past_forecast_days_naive_to_aware():
    client = OpenMeteoClient(
        forecast_url="test",
        historical_weather_url="test",
        historical_forecast_url="test",
        geocoding_url="test"
    )
    start = datetime.now()
    end = datetime.now()
    client._compute_past_forecast_days(start, end)

    past, forecast = client._compute_past_forecast_days(start, end)
    assert isinstance(past, int)
    assert isinstance(forecast, int)
