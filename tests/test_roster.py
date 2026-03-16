"""Tests for roster derivation."""
from unittest.mock import Mock
from f1pred.roster import (
    _entries_from_results,
    _same_round_known_roster,
    _latest_completed_round_in_season,
    _previous_completed_round_in_season,
    derive_roster,
)


def test_same_round_known_roster_qualifying():
    """Test that qualifying results are preferred if available."""
    jc = Mock()
    mock_q = [{"Driver": {"driverId": "q_driver"}, "Constructor": {}}]
    jc.get_qualifying_results.return_value = mock_q

    # Act
    roster = _same_round_known_roster(jc, "2024", "1")

    # Assert
    assert len(roster) == 1
    assert roster[0]["driverId"] == "q_driver"
    jc.get_qualifying_results.assert_called_once_with("2024", "1")
    jc.get_race_results.assert_not_called()
    jc.get_sprint_results.assert_not_called()


def test_same_round_known_roster_race_fallback():
    """Test that it falls back to race results if qualifying is unavailable/fails."""
    jc = Mock()
    jc.get_qualifying_results.side_effect = Exception("No Q")
    mock_r = [{"Driver": {"driverId": "r_driver"}, "Constructor": {}}]
    jc.get_race_results.return_value = mock_r

    # Act
    roster = _same_round_known_roster(jc, "2024", "1")

    # Assert
    assert len(roster) == 1
    assert roster[0]["driverId"] == "r_driver"
    jc.get_qualifying_results.assert_called_once_with("2024", "1")
    jc.get_race_results.assert_called_once_with("2024", "1")
    jc.get_sprint_results.assert_not_called()


def test_same_round_known_roster_sprint_fallback():
    """Test that it falls back to sprint results if Q and Race are unavailable."""
    jc = Mock()
    jc.get_qualifying_results.return_value = None
    jc.get_race_results.side_effect = Exception("No Race")
    mock_s = [{"Driver": {"driverId": "s_driver"}, "Constructor": {}}]
    jc.get_sprint_results.return_value = mock_s

    # Act
    roster = _same_round_known_roster(jc, "2024", "1")

    # Assert
    assert len(roster) == 1
    assert roster[0]["driverId"] == "s_driver"
    jc.get_sprint_results.assert_called_once_with("2024", "1")


def test_same_round_known_roster_none():
    """Test that it returns empty list if all fail."""
    jc = Mock()
    jc.get_qualifying_results.return_value = None
    jc.get_race_results.return_value = None
    jc.get_sprint_results.return_value = None

    # Act
    roster = _same_round_known_roster(jc, "2024", "1")

    # Assert
    assert roster == []


def test_latest_completed_round_in_season_race():
    """Test that _latest_completed_round_in_season prefers race results."""
    jc = Mock()
    # Schedule with 3 rounds
    jc.get_season_schedule.return_value = [
        {"round": "1"}, {"round": "2"}, {"round": "3"}
    ]
    # No results for round 3
    # Round 2 has race results
    def mock_get_race(s, r):
        if r == "2":
            return [{"Driver": {"driverId": "r2"}}]
        return None
    jc.get_race_results.side_effect = mock_get_race
    jc.get_qualifying_results.return_value = None

    rnd = _latest_completed_round_in_season(jc, "2024")
    assert rnd == "2"


def test_latest_completed_round_in_season_qualifying():
    """Test that _latest_completed_round_in_season falls back to qualifying results."""
    jc = Mock()
    # Schedule with 3 rounds
    jc.get_season_schedule.return_value = [
        {"round": "1"}, {"round": "2"}, {"round": "3"}
    ]
    jc.get_race_results.return_value = None
    # Round 3 has qualifying results but no race results yet
    def mock_get_qualifying(s, r):
        if r == "3":
            return [{"Driver": {"driverId": "q3"}}]
        return None
    jc.get_qualifying_results.side_effect = mock_get_qualifying

    rnd = _latest_completed_round_in_season(jc, "2024")
    assert rnd == "3"


def test_previous_completed_round_in_season():
    """Test that _previous_completed_round_in_season scans backward."""
    jc = Mock()
    # We are asking for previous to round 4
    # Round 3 has no results, Round 2 has results
    def mock_get_race(s, r):
        if r == "2":
            return [{"Driver": {"driverId": "r2"}}]
        return None
    jc.get_race_results.side_effect = mock_get_race
    jc.get_qualifying_results.return_value = None

    rnd = _previous_completed_round_in_season(jc, "2024", "4")
    assert rnd == "2"


def test_previous_completed_round_in_season_invalid_round():
    """Test that _previous_completed_round_in_season handles invalid int conversion."""
    jc = Mock()
    rnd = _previous_completed_round_in_season(jc, "2024", "invalid")
    assert rnd is None


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
