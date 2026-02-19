
import pytest
from colorama import Fore, Style
from f1pred.predict import _get_pos_color

def test_get_pos_color_gold():
    assert _get_pos_color(1) == Fore.YELLOW + Style.BRIGHT

def test_get_pos_color_silver():
    assert _get_pos_color(2) == Fore.WHITE + Style.BRIGHT

def test_get_pos_color_bronze():
    assert _get_pos_color(3) == Fore.MAGENTA + Style.BRIGHT

def test_get_pos_color_others():
    assert _get_pos_color(4) == Fore.RESET + Style.DIM
    assert _get_pos_color(10) == Fore.RESET + Style.DIM
    assert _get_pos_color(20) == Fore.RESET + Style.DIM
