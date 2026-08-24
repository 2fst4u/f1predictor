
from unittest.mock import patch, MagicMock
import f1pred.predict

# Direct sanitize_for_console behaviour (ANSI/newline/control-char handling) is
# covered comprehensively by tests/test_security_sanitization.py. This file
# keeps the integration test below, which proves run_predictions_for_event
# sanitises exception messages *before* they reach the logger.

@patch("f1pred.predict.logger")
def test_predict_logs_sanitized_exception(mock_logger):
    """Verify that run_predictions_for_event sanitizes exceptions before logging."""

    # We need to trigger an exception inside the session loop in predict.py
    # We mock resolve_event to return valid info, then mock build_session_features to raise

    with patch("f1pred.predict.resolve_event") as mock_resolve:
        mock_resolve.return_value = (2025, 1, {"raceName": "Test GP", "date": "2025-01-01", "time": "12:00:00Z"})

        with patch("f1pred.features.build_roster") as mock_roster:
            # Raise a malicious exception early in the loop
            malicious_msg = "Crash\n\033[31mForged\033[0m"
            mock_roster.side_effect = Exception(malicious_msg)

            # Mock other dependencies to avoid networking
            with patch("f1pred.predict.JolpicaClient"), \
                 patch("f1pred.predict.OpenMeteoClient"), \
                 patch("f1pred.predict.ensure_dirs"), \
                 patch("f1pred.predict.init_fastf1"), \
                 patch("f1pred.features.collect_historical_results", return_value=MagicMock(empty=False, __len__=lambda s: 1)), \
                 patch("f1pred.calibrate.CalibrationManager") as mock_cm, \
                 patch("f1pred.predict.StatusSpinner"):

                # Setup CalibrationManager mock
                mock_cm_instance = MagicMock()
                mock_cm_instance.check_calibration_needed.return_value = False
                mock_cm_instance.load_weights.return_value = {}
                mock_cm.return_value = mock_cm_instance

                # We mock _filter_sessions_for_round to return one session
                with patch("f1pred.predict._filter_sessions_for_round", return_value=["race"]):
                     f1pred.predict.run_predictions_for_event(
                        MagicMock(), # cfg
                        "2025",
                        "1",
                        ["race"]
                    )

            # Verify logger was called with sanitized message. Session failures
            # are logged at WARNING (they are the difference between a populated
            # and an empty prediction), so that is where the payload would land.
            found = False
            logged_messages = []
            for call in mock_logger.warning.call_args_list:
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


@patch("f1pred.predict.logger")
def test_predict_reports_failed_session_in_results(mock_logger):
    """A session that raises must be reported back to the caller.

    Without this the API stores an event with zero sessions and the UI renders
    a blank page, with no way to tell a failed run from "nothing to predict".
    """
    with patch("f1pred.predict.resolve_event") as mock_resolve:
        mock_resolve.return_value = (2025, 1, {"raceName": "Test GP", "date": "2025-01-01", "time": "12:00:00Z"})

        with patch("f1pred.features.build_roster") as mock_roster:
            mock_roster.side_effect = RuntimeError("upstream unavailable")

            with patch("f1pred.predict.JolpicaClient"), \
                 patch("f1pred.predict.OpenMeteoClient"), \
                 patch("f1pred.predict.ensure_dirs"), \
                 patch("f1pred.predict.init_fastf1"), \
                 patch("f1pred.calibrate.CalibrationManager") as mock_cm, \
                 patch("f1pred.predict.StatusSpinner"):

                mock_cm_instance = MagicMock()
                mock_cm_instance.check_calibration_needed.return_value = False
                mock_cm_instance.load_weights.return_value = {}
                mock_cm.return_value = mock_cm_instance

                with patch("f1pred.predict._filter_sessions_for_round", return_value=["race"]):
                    results = f1pred.predict.run_predictions_for_event(
                        MagicMock(), "2025", "1", ["race"], return_results=True
                    )

    assert results["sessions"] == {}
    assert "race" in results["errors"]
    assert "upstream unavailable" in results["errors"]["race"]
    assert "RuntimeError" in results["errors"]["race"]
