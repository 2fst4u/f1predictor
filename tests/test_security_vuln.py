import pytest
import time
from unittest.mock import patch, MagicMock
from f1pred.util import sanitize_for_console, StatusSpinner
from f1pred.data.jolpica import JolpicaClient

def test_sanitize_for_console_removes_ansi_codes():
    # Input with ANSI codes (red text)
    malicious_input = "\033[31mMalicious\033[0m"
    sanitized = sanitize_for_console(malicious_input)
    assert sanitized == "Malicious"

    # Input with simple text
    assert sanitize_for_console("Normal") == "Normal"

    # Mixed content
    mixed = "Hello \033[1mBold\033[0m World"
    assert sanitize_for_console(mixed) == "Hello Bold World"

    # Text that looks like ANSI but isn't quite (should pass through or be handled)
    # The regex \x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]) covers standard escape sequences.
    # We want to ensure it doesn't delete random text.
    safe_ish = "Price: $100 [USD]"
    assert sanitize_for_console(safe_ish) == safe_ish

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
    # 1. Test with a reasonable Retry-After value (should be respected)
    headers_ok = {"Retry-After": "120"}
    val_ok = JolpicaClient._retry_after_seconds(headers_ok, attempt=0, base=1.0, cap=30.0)
    # Currently, it respects it fully. After fix, it should still be 120 (if we cap at say 300)
    # or if we cap at 30, it will be 30.
    # The current code returns 120.

    # 2. Test with a MALICIOUS Retry-After value (e.g. 10 years)
    headers_bad = {"Retry-After": "315360000"}
    val_bad = JolpicaClient._retry_after_seconds(headers_bad, attempt=0, base=1.0, cap=30.0)

    # Assert behavior.
    # BEFORE FIX: val_bad would be 315360000.0
    # AFTER FIX: val_bad should be capped (e.g. 300.0)

    # We assert that we are implementing a cap.
    # Let's say we decide the cap is 300 seconds (5 minutes).
    MAX_ALLOWED_SLEEP = 300.0

    # This assertion is expected to FAIL before the fix if the code just returns `secs`.
    # But since I can't verify failure in this environment easily (I just want to implement the fix),
    # I will assert the SAFE behavior I want to see.
    assert val_bad <= MAX_ALLOWED_SLEEP, f"Retry-After value {val_bad} exceeded safety cap of {MAX_ALLOWED_SLEEP}"
