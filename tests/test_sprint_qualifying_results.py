import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from f1pred.predict import _get_actual_positions_for_session

def test_sprint_qualifying_results_retrieval_success():
    """
    Verify that 'sprint_qualifying' results are correctly retrieved
    using the new fallback logic that checks both names.
    """
    jc = MagicMock()
    roster_view = pd.DataFrame([
        {"driverId": "max_verstappen", "number": "1", "code": "VER"},
        {"driverId": "lando_norris", "number": "4", "code": "NOR"}
    ])

    with patch("f1pred.predict.get_session_classification") as mock_get_cls:
        def side_effect(season, round_no, session_name):
            if session_name == "Sprint Qualifying":
                return pd.DataFrame([
                    {"Abbreviation": "VER", "DriverNumber": 1, "Position": 1},
                    {"Abbreviation": "NOR", "DriverNumber": 4, "Position": 2}
                ])
            return None

        mock_get_cls.side_effect = side_effect

        # Call with sprint_qualifying
        results = _get_actual_positions_for_session(jc, 2026, 4, "sprint_qualifying", roster_view)

        assert results is not None
        assert results.iloc[0] == 1
        assert results.iloc[1] == 2

def test_sprint_qualifying_mapping_robustness():
    """
    Test that mapping succeeds via Abbreviation if DriverNumber is mismatched.
    """
    jc = MagicMock()
    # Roster has correct codes but different numbers
    roster_view = pd.DataFrame([
        {"driverId": "max_verstappen", "number": "99", "code": "VER"},
        {"driverId": "lando_norris", "number": "88", "code": "NOR"}
    ])

    # FastF1 has results with correct Abbreviations but different DriverNumbers
    mock_cls = pd.DataFrame([
        {"Abbreviation": "VER", "DriverNumber": 1, "Position": 1},
        {"Abbreviation": "NOR", "DriverNumber": 4, "Position": 2}
    ])

    with patch("f1pred.predict.get_session_classification", return_value=mock_cls):
        results = _get_actual_positions_for_session(jc, 2026, 4, "sprint_qualifying", roster_view)

        # Should succeed via Abbreviation fallback
        assert results is not None
        assert not results.isna().all()
        assert results.iloc[0] == 1
        assert results.iloc[1] == 2

def test_sprint_qualifying_fuzzy_name_match():
    """
    Test that mapping succeeds via fuzzy name match if others fail.
    """
    jc = MagicMock()
    # Roster has names but mismatched numbers/codes
    roster_view = pd.DataFrame([
        {"driverId": "max_verstappen", "number": "99", "code": "XXX", "name": "Max Verstappen"},
        {"driverId": "lando_norris", "number": "88", "code": "YYY", "name": "Lando Norris"}
    ])

    # FastF1 has results with correct Names but different details
    mock_cls = pd.DataFrame([
        {"LastName": "Verstappen", "Position": 1},
        {"LastName": "Norris", "Position": 2}
    ])

    with patch("f1pred.predict.get_session_classification", return_value=mock_cls):
        results = _get_actual_positions_for_session(jc, 2026, 4, "sprint_qualifying", roster_view)

        # Should succeed via Fuzzy Name fallback
        assert results is not None
        assert not results.isna().all()
        assert results.iloc[0] == 1
        assert results.iloc[1] == 2
