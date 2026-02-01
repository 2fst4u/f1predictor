import pytest
from colorama import Fore, Style
from f1pred.predict import _render_actual_pos

def test_render_actual_pos_exact_match():
    # Predicted 1, Actual 1
    # Expect: Green color, checkmark
    # width 6. "✓" + "1" = 2 chars. Padding 4.
    result = _render_actual_pos(1, 1, width=6)
    # We check exact string construction
    expected = f"    {Fore.GREEN}{Style.BRIGHT}✓1{Style.RESET_ALL}"
    assert result == expected

def test_render_actual_pos_close_match():
    # Predicted 1, Actual 2 (Diff 1)
    # Expect: Cyan color, no symbol
    # width 6. "2" = 1 char. Padding 5.
    result = _render_actual_pos(1, 2, width=6)
    expected = f"     {Fore.CYAN}{Style.BRIGHT}2{Style.RESET_ALL}"
    assert result == expected

def test_render_actual_pos_ok_match():
    # Predicted 1, Actual 5 (Diff 4)
    # Expect: Yellow color
    result = _render_actual_pos(1, 5, width=6)
    expected = f"     {Fore.YELLOW}{Style.BRIGHT}5{Style.RESET_ALL}"
    assert result == expected

def test_render_actual_pos_bad_match():
    # Predicted 1, Actual 20 (Diff 19)
    # Expect: Red color
    # width 6. "20" = 2 chars. Padding 4.
    result = _render_actual_pos(1, 20, width=6)
    expected = f"    {Fore.RED}{Style.BRIGHT}20{Style.RESET_ALL}"
    assert result == expected

def test_render_actual_pos_custom_width():
    # Width 4
    # Exact match: "✓1" = 2 chars. Padding 2.
    result = _render_actual_pos(1, 1, width=4)
    expected = f"  {Fore.GREEN}{Style.BRIGHT}✓1{Style.RESET_ALL}"
    assert result == expected
