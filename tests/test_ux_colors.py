import pytest
from colorama import Fore, Style
from f1pred.predict import _get_team_color

@pytest.mark.parametrize("team_name, expected_color", [
    ("Ferrari", Fore.RED),
    ("Scuderia Ferrari", Fore.RED),
    ("Red Bull Racing", Fore.BLUE),
    ("Oracle Red Bull Racing", Fore.BLUE),
    ("Mercedes", Fore.CYAN),
    ("Mercedes-AMG PETRONAS F1 Team", Fore.CYAN),
    ("McLaren", Fore.YELLOW),
    ("McLaren F1 Team", Fore.YELLOW),
    ("Aston Martin", Fore.GREEN),
    ("Aston Martin Aramco F1 Team", Fore.GREEN),
    ("Alpine", Fore.MAGENTA),
    ("BWT Alpine F1 Team", Fore.MAGENTA),
    ("Williams", Fore.BLUE + Style.BRIGHT),
    ("Williams Racing", Fore.BLUE + Style.BRIGHT),
    ("Haas F1 Team", Fore.WHITE),
    ("MoneyGram Haas F1 Team", Fore.WHITE),
    ("Kick Sauber", Fore.GREEN + Style.BRIGHT),
    ("Stake F1 Team Kick Sauber", Fore.GREEN + Style.BRIGHT),
    ("RB", Fore.BLUE),
    ("Visa Cash App RB F1 Team", Fore.BLUE),
    ("AlphaTauri", Fore.BLUE), # Legacy support
    ("Unknown Team", Style.DIM),
    ("", Style.DIM),
    (None, Style.DIM),
])
def test_get_team_color(team_name, expected_color):
    assert _get_team_color(team_name) == expected_color
