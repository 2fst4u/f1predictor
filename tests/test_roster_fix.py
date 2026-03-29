import pytest
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from f1pred.roster import derive_roster

def test_roster_derivation_prefers_later_sessions():
    """
    Reproduce the issue where FP1 roster is picked even if Qualifying results are available.
    In 2026 Japan GP, Jack Crawford did FP1 but Fernando Alonso did Qualifying and Race.
    """
    jc = Mock()
    # Ensure Jolpica returns nothing for the current round to reach FastF1 logic
    jc.get_qualifying_results.return_value = []
    jc.get_race_results.return_value = []
    jc.get_sprint_results.return_value = []
    jc.get_season_entry_list.return_value = []
    jc.get_latest_season_and_round.return_value = ("2025", "22")

    # Mock FastF1 event and sessions
    mock_event = Mock()

    # Mock FP1 session
    fp1_session = Mock()
    fp1_session.results = pd.DataFrame([
        {
            "Abbreviation": "CRA",
            "DriverNumber": "31",
            "FirstName": "Jack",
            "LastName": "Crawford",
            "TeamName": "Aston Martin"
        }
    ])

    # Mock Qualifying session
    quali_session = Mock()
    quali_session.results = pd.DataFrame([
        {
            "Abbreviation": "ALO",
            "DriverNumber": "14",
            "FirstName": "Fernando",
            "LastName": "Alonso",
            "TeamName": "Aston Martin"
        }
    ])

    def get_session_mock(name):
        if name == "FP1":
            return fp1_session
        if name == "Qualifying":
            return quali_session
        # Return empty session for others
        s = Mock()
        s.results = pd.DataFrame()
        return s

    mock_event.get_session.side_effect = get_session_mock

    with patch("f1pred.data.fastf1_backend.get_event", return_value=mock_event):
        # We need to mock _get_canonical_mapping to return something that won't fail
        # Or just let it return empty dict, and it will use Abbreviations.

        roster = derive_roster(jc, "2026", "4")

        # Currently, it picks FP1 first because it iterates ["FP1", "Qualifying", ...]
        # and breaks at the first hit.
        driver_ids = [d["driverId"] for d in roster]

        # REPRODUCTION: This assertion currently fails if we expect ALO, but passes if we got CRA
        # The user says Jack Crawford (CRA) was incorrectly derived.
        assert "alo" in driver_ids
        assert "cra" not in driver_ids
