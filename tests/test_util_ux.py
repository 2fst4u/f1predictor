import sys
import time
import pytest
from unittest.mock import MagicMock, call, patch
from colorama import Style
from f1pred.util import print_countdown, HIDE_CURSOR, SHOW_CURSOR

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
