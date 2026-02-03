import pytest
from colorama import Fore
from f1pred.predict import _render_actual_pos

def test_render_actual_pos_exact():
    # Diff 0 -> Green, Checkmark
    output = _render_actual_pos(1, 1)
    assert "✓" in output
    assert Fore.GREEN in output
    assert "1" in output

def test_render_actual_pos_close():
    # Diff 2 -> Cyan, Approx symbol
    output = _render_actual_pos(3, 1)
    assert "≈" in output
    assert Fore.CYAN in output
    assert "1" in output

def test_render_actual_pos_medium():
    # Diff 5 -> Yellow, No symbol
    output = _render_actual_pos(6, 1)
    assert "≈" not in output
    assert "✓" not in output
    assert Fore.YELLOW in output
    assert "1" in output

def test_render_actual_pos_far():
    # Diff 10 -> Red, No symbol
    output = _render_actual_pos(11, 1)
    assert "≈" not in output
    assert "✓" not in output
    assert Fore.RED in output
    assert "1" in output
