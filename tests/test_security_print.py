
import pytest
import pandas as pd
from unittest.mock import MagicMock
from f1pred.predict import print_session_console
from colorama import Fore, Style

def test_print_session_console_sanitization(capsys):
    # Mock data
    df = pd.DataFrame({
        "predicted_position": [1],
        "name": ["Max Verstappen"],
        "code": ["VER"],
        "constructorName": ["Red Bull"],
        "mean_pos": [1.0],
        "p_top3": [0.9],
        "p_win": [0.8],
        "p_dnf": [0.1],
        "grid": [1],
        "actual_position": [1]
    })

    cfg = MagicMock()

    # Malicious circuit name with ANSI codes (Terminal Escape Injection)
    malicious_circuit = f"Spa-Francorchamps{Fore.RED}[INJECTED]{Style.RESET_ALL}"

    print_session_console(
        df,
        "race",
        cfg,
        circuit_name=malicious_circuit
    )

    captured = capsys.readouterr()

    print("\nCaptured Output:")
    print(repr(captured.out))

    # ASSERT THE FIX:
    # 1. The ANSI escape code \x1b[31m (RED) should be stripped.
    # 2. The malicious payload text might remain if we only strip ANSI,
    #    but sanitize_for_console handles ANSI codes.
    #    Wait, sanitize_for_console removes ANSI codes but keeps the text.
    #    So "[INJECTED]" will be present, but it won't be RED.

    # Check that NO ANSI codes from the malicious input are present around [INJECTED]
    # The output will still contain ANSI codes from the print_session_console function itself (yellow headers etc).

    # Let's inspect what sanitize_for_console does to `malicious_circuit`:
    # It should become "Spa-Francorchamps[INJECTED]"

    assert "Spa-Francorchamps[INJECTED]" in captured.out

    # We want to ensure that the SPECIFIC malicious ANSI sequence is gone.
    # malicious_circuit contained Fore.RED which is \x1b[31m

    # However, the code adds " | " before it.
    # And there are other colors in the output.

    # Let's construct the expected clean string for the header part
    # header_line = f"\n{Fore.YELLOW}{Style.BRIGHT}== {title}"
    # header_line += f" | {sanitize_for_console(circuit_name)}"
    # header_line += f" =={Style.RESET_ALL}"

    # So we expect: ... | Spa-Francorchamps[INJECTED] == ...
    # And NOT: ... | Spa-Francorchamps\x1b[31m[INJECTED]\x1b[0m == ...

    assert "| Spa-Francorchamps[INJECTED] ==" in captured.out
    assert "| Spa-Francorchamps\x1b[31m[INJECTED]" not in captured.out

if __name__ == "__main__":
    pytest.main([__file__])
