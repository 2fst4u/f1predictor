"""Performance tests for roster derivation."""
import pytest
from unittest.mock import MagicMock, patch
from f1pred.roster import derive_roster
from f1pred.data.jolpica import JolpicaClient

class MockJolpicaClient:
    def __init__(self):
        self.call_count = 0
        self.get_latest_calls = 0

    def get_qualifying_results(self, season, rnd):
        self.call_count += 1
        return []

    def get_race_results(self, season, rnd):
        self.call_count += 1
        return []

    def get_sprint_results(self, season, rnd):
        self.call_count += 1
        return []

    def get_season_schedule(self, season):
        self.call_count += 1
        # Return a fake schedule with 24 rounds
        return [{"round": str(i)} for i in range(1, 25)]

    def get_latest_season_and_round(self):
        self.get_latest_calls += 1
        return ("2024", "20")

def test_future_season_optimization():
    """Verify that predicting for a future season triggers the optimization."""
    jc = MockJolpicaClient()

    # Patch _roster_from_fastf1 to return [] so we fall through to fallback logic
    with patch("f1pred.roster._roster_from_fastf1", return_value=[]):
        derive_roster(jc, "2026", "1")

    # Expected calls:
    # 1. check current round (qual/race/sprint) = 3
    # 2. get_latest_season_and_round = 1
    # 3. get fallback roster (race or qual) = 1
    # Total = 5
    # Without optimization, it would be > 100
    assert jc.get_latest_calls == 1
    assert jc.call_count <= 10, f"Too many API calls: {jc.call_count}"

def test_future_round_optimization():
    """Verify that predicting for a future round in current season triggers optimization."""
    jc = MockJolpicaClient()

    # Patch _roster_from_fastf1 to return []
    with patch("f1pred.roster._roster_from_fastf1", return_value=[]):
        # Requesting 2024 round 24 (future, since latest is 20)
        derive_roster(jc, "2024", "24")

    # Expected calls:
    # 1. check current round (qual/race/sprint) = 3
    # 2. get_latest_season_and_round = 1
    # 3. get fallback roster (race or qual) = 1
    # Total = 5
    assert jc.get_latest_calls == 1
    assert jc.call_count <= 10, f"Too many API calls: {jc.call_count}"
