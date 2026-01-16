import pytest
from f1pred.data.jolpica import JolpicaClient

def test_validate_season_log_injection_prevention():
    """
    Test that _validate_season prevents log injection by escaping
    control characters in the exception message.
    """
    jc = JolpicaClient("http://mock", timeout=1)

    # Input with newline (Log Injection / Forging attempt)
    malicious_input = "2025\n[CRITICAL] System compromised"

    with pytest.raises(ValueError) as excinfo:
        jc._validate_season(malicious_input)

    msg = str(excinfo.value)
    # The raw newline must NOT be present
    assert "\n" not in msg
    # The escaped newline (from repr) should be present
    assert "\\n" in msg
    # The input should be quoted (repr style)
    assert "'2025\\n[CRITICAL] System compromised'" in msg or '"2025\\n[CRITICAL] System compromised"' in msg

def test_validate_round_log_injection_prevention():
    """
    Test that _validate_round prevents log injection by escaping
    control characters in the exception message.
    """
    jc = JolpicaClient("http://mock", timeout=1)
    malicious_input = "1\nINFO: Fake Log"

    with pytest.raises(ValueError) as excinfo:
        jc._validate_round(malicious_input)

    msg = str(excinfo.value)
    assert "\n" not in msg
    assert "\\n" in msg
    assert "1\\nINFO: Fake Log" in msg
