import pytest
from f1pred.util import sanitize_for_console

def test_sanitize_removes_ansi_codes():
    """Verify ANSI escape codes are stripped."""
    assert sanitize_for_console("\033[31mRed\033[0m") == "Red"

def test_sanitize_replaces_whitespace_controls():
    """Verify newlines, tabs, etc. are replaced by space."""
    # Carriage Return -> space (prevent overwrite)
    assert sanitize_for_console("Failed\rSuccess") == "Failed Success"
    # Newline -> space (prevent log forging)
    assert sanitize_for_console("Line1\nLine2") == "Line1 Line2"
    # Tab -> space (prevent alignment issues)
    assert sanitize_for_console("Col1\tCol2") == "Col1 Col2"

def test_sanitize_removes_other_controls():
    """Verify dangerous control characters are removed."""
    # Backspace (hide chars)
    assert sanitize_for_console("Dangerous\bSafe") == "DangerousSafe"
    # Bell
    assert sanitize_for_console("Ding\aDong") == "DingDong"
    # Null
    assert sanitize_for_console("Null\0Value") == "NullValue"

def test_sanitize_preserves_unicode():
    """Verify unicode characters (like accents) are preserved."""
    assert sanitize_for_console("Pérez") == "Pérez"
    assert sanitize_for_console("Räikkönen") == "Räikkönen"
    assert sanitize_for_console("Tsunoda (角田)") == "Tsunoda (角田)"

def test_sanitize_mixed_injection():
    """Verify handling of mixed ANSI and control chars."""
    malicious = "Error\r\033[31mForged\033[0m\nLog"
    # \r -> space, ANSI -> removed, \n -> space
    # "Error Forged Log"
    assert sanitize_for_console(malicious) == "Error Forged Log"
