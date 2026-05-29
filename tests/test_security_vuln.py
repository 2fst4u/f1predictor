from datetime import datetime, timezone, timedelta
from email.utils import format_datetime
from unittest.mock import patch
from f1pred.util import StatusSpinner
from f1pred.data.jolpica import JolpicaClient
from f1pred.data.open_meteo import OpenMeteoClient

# Note: direct sanitize_for_console behaviour is covered comprehensively by
# tests/test_security_sanitization.py. This file keeps only the integration-style
# checks (StatusSpinner) that exercise sanitisation through a real call path.

def test_status_spinner_sanitizes_message(capsys):
    # Verify StatusSpinner automatically sanitizes its input message
    malicious_input = "\033[31mMalicious\033[0m"
    with StatusSpinner(f"Processing {malicious_input}"):
        pass

    captured = capsys.readouterr()
    # The output should NOT contain the escape code
    assert "\033[31m" not in captured.out
    # But it should contain the clean text
    assert "Processing Malicious" in captured.out

def test_jolpica_retry_after_dos_prevention():
    """
    Test that the Jolpica client caps the 'Retry-After' header value
    to prevent a Denial of Service (DoS) where a malicious or broken API
    forces the application to sleep for an excessive amount of time.
    """
    MAX_ALLOWED_SLEEP = 300.0  # the hard cap implemented in _retry_after_seconds

    # 1. A reasonable numeric Retry-After (under the cap) is respected exactly.
    val_ok = JolpicaClient._retry_after_seconds(
        {"Retry-After": "120"}, attempt=0, base=1.0, cap=30.0)
    assert val_ok == 120.0

    # 2. A malicious numeric Retry-After (10 years) is capped, not honoured.
    val_bad = JolpicaClient._retry_after_seconds(
        {"Retry-After": "315360000"}, attempt=0, base=1.0, cap=30.0)
    assert val_bad == MAX_ALLOWED_SLEEP

    # 3. An HTTP-date far in the future is also capped (covers the date branch).
    future = format_datetime(datetime.now(timezone.utc) + timedelta(days=365))
    val_date = JolpicaClient._retry_after_seconds(
        {"Retry-After": future}, attempt=0, base=1.0, cap=30.0)
    assert val_date <= MAX_ALLOWED_SLEEP

def test_open_meteo_timezone_validation():
    """Test that OpenMeteoClient validates timezone input preventing potential injection or errors."""
    client = OpenMeteoClient(
        forecast_url="http://mock",
        historical_weather_url="http://mock-hist",
        historical_forecast_url="http://mock-hist-forecast",
        geocoding_url="http://mock-geo",
    )

    start = datetime(2023, 1, 1)
    end = datetime(2023, 1, 2)
    bad_tz = "Europe/London\nnewline" # potentially malicious input

    with patch("f1pred.data.open_meteo.http_get_json") as mock_get:
        mock_get.return_value = {}

        # Test get_historical_weather
        client.get_historical_weather(0, 0, start, end, tz=bad_tz)
        args, kwargs = mock_get.call_args
        params = kwargs.get('params', {})
        assert params.get('timezone') == "UTC", "Should fallback to UTC for invalid timezone"

        # Test get_historical_forecast
        client.get_historical_forecast(0, 0, start, end, tz=bad_tz)
        args, kwargs = mock_get.call_args
        params = kwargs.get('params', {})
        assert params.get('timezone') == "UTC", "Should fallback to UTC for invalid timezone"

        # Test valid timezone
        client.get_historical_weather(0, 0, start, end, tz="Europe/London")
        args, kwargs = mock_get.call_args
        params = kwargs.get('params', {})
        assert params.get('timezone') == "Europe/London", "Should accept valid timezone"
