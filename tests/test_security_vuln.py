import pytest
from f1pred.util import sanitize_for_console, StatusSpinner

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
