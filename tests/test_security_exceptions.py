
import pytest
from unittest.mock import patch, MagicMock
from f1pred.util import sanitize_for_console
import f1pred.predict

def test_sanitize_exception_message():
    """Verify that sanitize_for_console effectively neutralizes malicious exception messages."""

    # 1. Newlines and Carriage Returns (Log Forging / Terminal Spoofing)
    malicious_msg = "Error occurred\r\n[INFO] Forged Log Entry"
    clean = sanitize_for_console(malicious_msg)
    assert "\r" not in clean
    assert "\n" not in clean
    assert "Forged Log Entry" in clean
    # Expect spaces instead (2 spaces for \r and \n)
    assert "Error occurred  [INFO] Forged Log Entry" == clean

    # 2. ANSI Codes (Terminal Injection)
    ansi_msg = "\033[31mCritical Error\033[0m"
    clean_ansi = sanitize_for_console(ansi_msg)
    assert "\033" not in clean_ansi
    assert "Critical Error" == clean_ansi

    # 3. Mixed
    mixed = "Error\n\033[31mHacked\033[0m"
    clean_mixed = sanitize_for_console(mixed)
    # \n -> space, ANSI removed. Result: "Error Hacked"
    assert clean_mixed == "Error Hacked"

@patch("f1pred.predict.logger")
def test_predict_logs_sanitized_exception(mock_logger):
    """Verify that run_predictions_for_event sanitizes exceptions before logging."""

    # We need to trigger an exception inside the loop in predict.py
    # We can mock resolve_event to return valid info, then mock build_session_features to raise

    with patch("f1pred.predict.resolve_event") as mock_resolve:
        mock_resolve.return_value = (2025, 1, {"raceName": "Test GP", "date": "2025-01-01", "time": "12:00:00Z"})

        with patch("f1pred.predict.build_session_features") as mock_build:
            # Raise a malicious exception
            malicious_msg = "Crash\n\033[31mForged\033[0m"
            mock_build.side_effect = Exception(malicious_msg)

            # Mock other dependencies to avoid networking
            with patch("f1pred.predict.JolpicaClient"), \
                 patch("f1pred.predict.OpenMeteoClient"), \
                 patch("f1pred.predict.ensure_dirs"), \
                 patch("f1pred.predict.init_fastf1"):

                # Run prediction
                # We mock _filter_sessions_for_round to return one session
                with patch("f1pred.predict._filter_sessions_for_round", return_value=["race"]):
                     f1pred.predict.run_predictions_for_event(
                        MagicMock(), # cfg
                        "2025",
                        "1",
                        ["race"]
                    )

            # Verify logger was called with sanitized message
            # The logger.info is called multiple times. We look for the one with the exception.
            # Expected: logger.info("Exception: Crash Forged") or similar
            # Note: "Crash Hacked" logic above suggests 1 space.

            found = False
            logged_messages = []
            for call in mock_logger.info.call_args_list:
                args, _ = call
                msg = args[0]
                logged_messages.append(msg)
                # We look for the sanitized string "Crash Forged"
                if "Crash Forged" in msg:
                    found = True
                    # Ensure raw payload is NOT present
                    assert "\n" not in msg
                    assert "\033" not in msg
                    break

            assert found, f"Logger did not receive sanitized exception message. Logged: {logged_messages}"
