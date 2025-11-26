"""Tests for roster derivation."""
import pytest
from unittest.mock import Mock, MagicMock
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
    jc.get_qualifying_results.return_value = [
        {
            'Driver': {'driverId': 'verstappen', 'code': 'VER'},
            'Constructor': {'constructorId': 'red_bull', 'name': 'Red Bull'},
        },
    ]

    event_dt = datetime(2023, 3, 5, tzinfo=timezone.utc)
    now_dt = datetime(2023, 3, 10, tzinfo=timezone.utc)

    roster = derive_roster(jc, '2023', '1', event_dt=event_dt, now_dt=now_dt)

    assert len(roster) == 1
    assert roster[0]['driverId'] == 'verstappen'
