import sys
import time
import logging
import pytest
from unittest.mock import MagicMock, call, patch
from colorama import Fore, Style
from f1pred.util import print_countdown, HIDE_CURSOR, SHOW_CURSOR, StatusSpinner, safe_float

@patch("f1pred.util.sys.stdout")
@patch("f1pred.util.time.sleep")
def test_print_countdown_tty(mock_sleep, mock_stdout):
    # Setup mock to return True for isatty
    mock_stdout.isatty.return_value = True

    # Run for 2 seconds
    print_countdown(2, "Test")

    # Verify cursor hiding
    assert mock_stdout.write.call_args_list[0] == call(HIDE_CURSOR)

    # Verify loop writes
    # Loop range(2, 0, -1) -> 2, 1

    # We expect these calls:
    # 1. HIDE_CURSOR
    # 2. \r ... 2s ...
    # 3. \r ... 1s ...
    # 4. \r\033[K (clear line)
    # 5. SHOW_CURSOR (finally block)

    expected_calls = [
        call(HIDE_CURSOR),
        call(f"\r{Style.DIM}↻ Test 2s...{Style.RESET_ALL}\033[K"),
        call(f"\r{Style.DIM}↻ Test 1s...{Style.RESET_ALL}\033[K"),
        call("\r\033[K"),
        call(SHOW_CURSOR)
    ]

    # Filter only write calls
    mock_stdout.write.assert_has_calls(expected_calls, any_order=False)

    # Verify sleep calls
    # sleep(1) called 2 times
    assert mock_sleep.call_count == 2
    mock_sleep.assert_called_with(1)

@patch("f1pred.util.sys.stdout")
@patch("f1pred.util.time.sleep")
def test_print_countdown_non_tty(mock_sleep, mock_stdout):
    # Setup mock to return False for isatty
    mock_stdout.isatty.return_value = False

    # Run for 5 seconds
    print_countdown(5, "Test")

    # Verify simple sleep
    mock_sleep.assert_called_once_with(5)

    # Verify no writes
    mock_stdout.write.assert_not_called()

def test_safe_float():
    assert safe_float("12.34") == 12.34
    assert safe_float(5) == 5.0
    assert safe_float("invalid") is None
    assert safe_float("invalid", default=0.0) == 0.0
    assert safe_float(None, default=-1.0) == -1.0

@patch("f1pred.util.sys.stdout")
@patch("f1pred.util.time.sleep")
def test_status_spinner_non_tty(mock_sleep, mock_stdout):
    mock_stdout.isatty.return_value = False

    with StatusSpinner("Test Non-TTY") as spinner:
        assert spinner.running is True
        assert spinner.thread is None  # Should not start thread
        spinner.update("New Message")
        assert spinner.message == "New Message"

    assert spinner.running is False
    # Should print success message at end
    # Note: simple print() writes to stdout, which is mocked
    # We check if write was called with expected success format
    # The print inside __exit__ uses Fore.GREEN etc.
    # It might call write multiple times or once depending on implementation details of print
    # But since we mock stdout, we can check calls.
    # Actually print() calls write(str) then write('\n')

    # We just want to ensure it didn't try to hide cursor or spin
    assert call(HIDE_CURSOR) not in mock_stdout.write.call_args_list

@patch("f1pred.util.sys.stdout")
@patch("f1pred.util.time.sleep")
def test_status_spinner_tty(mock_sleep, mock_stdout):
    mock_stdout.isatty.return_value = True

    with StatusSpinner("Test TTY", delay=0.01) as spinner:
        assert spinner.thread is not None
        assert spinner.thread.is_alive()
        # Let it spin a bit
        time.sleep(0.05)

    # Thread should be joined and dead
    assert not spinner.thread.is_alive()

    # Verify cursor hide/show
    assert call(HIDE_CURSOR) in mock_stdout.write.call_args_list
    assert call(SHOW_CURSOR) in mock_stdout.write.call_args_list

def test_status_spinner_logging_suppression():
    logger = logging.getLogger()
    original_level = logger.level

    # Ensure we start at INFO or lower to test suppression
    logger.setLevel(logging.INFO)

    with StatusSpinner("Log Test"):
        # Inside context, level should be WARNING
        assert logger.getEffectiveLevel() == logging.WARNING

    # After context, should restore to INFO
    assert logger.getEffectiveLevel() == logging.INFO

    logger.setLevel(original_level)

@patch("f1pred.util.sys.stdout")
def test_status_spinner_exit_states(mock_stdout):
    # Success state
    with StatusSpinner("Success Test") as s:
        s.set_status("success")
    # Verify green checkmark (approximately)
    # We look for "Success Test" in calls
    found_success = any("Success Test" in str(c) for c in mock_stdout.write.call_args_list)
    assert found_success

    # Skipped state
    with StatusSpinner("Skipped Test") as s:
        s.set_status("skipped")
    found_skipped = any("Skipped Test" in str(c) for c in mock_stdout.write.call_args_list)
    assert found_skipped

    # Exception state
    try:
        with StatusSpinner("Fail Test"):
            raise ValueError("Boom")
    except ValueError:
        pass
    found_fail = any("Fail Test" in str(c) and "(Failed)" in str(c) for c in mock_stdout.write.call_args_list)
    assert found_fail

def test_status_spinner_updates():
    with StatusSpinner("Initial") as s:
        assert s.message == "Initial"
        s.update("Updated")
        assert s.message == "Updated"

def test_status_spinner_callback():
    callback = MagicMock()
    with StatusSpinner("Initial", on_update=callback) as s:
        # Callback should be called on entry
        callback.assert_called_once_with("Initial")
        callback.reset_mock()

        s.update("Updated")
        callback.assert_called_once_with("Updated")
