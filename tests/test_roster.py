"""Tests for roster derivation."""
from unittest.mock import Mock
from f1pred.roster import (
    _entries_from_results,
    derive_roster,
)


def test_entries_from_results():
    """Test conversion of results to entry list."""
    results = [
        {
            'Driver': {
                'driverId': 'hamilton',
                'code': 'HAM',
                'givenName': 'Lewis',
                'familyName': 'Hamilton',
                'permanentNumber': '44',
            },
            'Constructor': {
                'constructorId': 'mercedes',
                'name': 'Mercedes',
            },
        },
        {
            'Driver': {
                'driverId': 'hamilton',  # Duplicate
                'code': 'HAM',
                'givenName': 'Lewis',
                'familyName': 'Hamilton',
                'permanentNumber': '44',
            },
            'Constructor': {
                'constructorId': 'mercedes',
                'name': 'Mercedes',
            },
        },
    ]

    entries = _entries_from_results(results)

    # Should deduplicate
    assert len(entries) == 1
    assert entries[0]['driverId'] == 'hamilton'
    assert entries[0]['constructorId'] == 'mercedes'


def test_derive_roster_past_event():
    """Test roster derivation for past events."""
    from datetime import datetime, timezone

    jc = Mock()
    # Provide 20 drivers to satisfy the completeness heuristic for modern seasons
    mock_results = [
        {
            'Driver': {'driverId': f'driver_{i}', 'code': f'D{i}'},
            'Constructor': {'constructorId': 'team', 'name': 'Team'},
        } for i in range(20)
    ]
    jc.get_qualifying_results.return_value = mock_results

    event_dt = datetime(2023, 3, 5, tzinfo=timezone.utc)
    now_dt = datetime(2023, 3, 10, tzinfo=timezone.utc)

    roster = derive_roster(jc, '2023', '1', event_dt=event_dt, now_dt=now_dt)

    assert len(roster) == 20
    assert roster[0]['driverId'] == 'driver_0'

def test_same_round_known_roster_exceptions():
    """Test same round known roster with API exceptions."""
    from f1pred.roster import _same_round_known_roster

    jc = Mock()
    jc.get_qualifying_results.side_effect = Exception("API error")
    jc.get_race_results.side_effect = Exception("API error")
    jc.get_sprint_results.side_effect = Exception("API error")

    res = _same_round_known_roster(jc, "2023", "1")
    assert res == []

def test_latest_completed_round_in_season():
    """Test getting the latest completed round in a season."""
    from f1pred.roster import _latest_completed_round_in_season

    jc = Mock()
    jc.get_season_schedule.return_value = [
        {"round": 1},
        {"round": 2},
        {"round": 3}
    ]

    def mock_race_results(season, rnd):
        if rnd == "1":
            return [{"Driver": {"driverId": "hamilton"}}]
        return None

    def mock_qualifying_results(season, rnd):
        if rnd == "2":
            return [{"Driver": {"driverId": "max"}}]
        return None

    jc.get_race_results.side_effect = mock_race_results
    jc.get_qualifying_results.side_effect = mock_qualifying_results

    # Should check 3 (None), then 2 (Qualifying) -> return "2"
    res = _latest_completed_round_in_season(jc, "2023")
    assert res == "2"

    # Test Exception in get_season_schedule
    jc.get_season_schedule.side_effect = Exception("API error")
    assert _latest_completed_round_in_season(jc, "2023") is None

def test_previous_completed_round_in_season():
    """Test getting the previous completed round in a season."""
    from f1pred.roster import _previous_completed_round_in_season

    jc = Mock()
    def mock_race_results(season, rnd):
        if rnd == "1":
            return [{"Driver": {"driverId": "hamilton"}}]
        return None

    def mock_qualifying_results(season, rnd):
        if rnd == "2":
            return [{"Driver": {"driverId": "max"}}]
        if rnd == "3":
            raise Exception("API Error")
        return None

    jc.get_race_results.side_effect = mock_race_results
    jc.get_qualifying_results.side_effect = mock_qualifying_results

    # Check from 4 downwards. 3 raises exception but continues, 2 returns qualifying -> "2"
    assert _previous_completed_round_in_season(jc, "2023", "4") == "2"

    # Check from 2 downwards. 1 returns race -> "1"
    assert _previous_completed_round_in_season(jc, "2023", "2") == "1"

    # Invalid upto_round
    assert _previous_completed_round_in_season(jc, "2023", "invalid") is None
