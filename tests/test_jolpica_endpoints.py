import unittest
from unittest.mock import MagicMock

import pytest

from f1pred.data.jolpica import JolpicaClient


# The three bulk season endpoints (race / qualifying / sprint) share an
# identical paginate-then-merge-by-round implementation; the only differences
# are the endpoint path and the per-round results key. Parametrising over that
# (method, key) pair exercises the merge logic for all three without three
# copy-pasted test bodies.
@pytest.mark.parametrize(
    "method_name, results_key",
    [
        ("get_season_race_results", "Results"),
        ("get_season_qualifying_results", "QualifyingResults"),
        ("get_season_sprint_results", "SprintResults"),
    ],
)
def test_get_season_bulk_results_merge_by_round(method_name, results_key):
    client = JolpicaClient("http://mock")
    # Round 1 appears on two pages (must be merged into 2 entries); round 2 on one.
    client._fetch_paginated_parallel = MagicMock(return_value=[
        {"RaceTable": {"Races": [{"round": "1", results_key: [{"position": "1"}]}]}},
        {"RaceTable": {"Races": [
            {"round": "1", results_key: [{"position": "2"}]},
            {"round": "2", results_key: [{"position": "1"}]},
        ]}},
    ])
    results = getattr(client, method_name)("2021")
    assert len(results) == 2

    round1 = next((r for r in results if r["round"] == "1"), None)
    assert round1 is not None
    assert len(round1[results_key]) == 2  # merged across pages

    round2 = next((r for r in results if r["round"] == "2"), None)
    assert round2 is not None
    assert len(round2[results_key]) == 1


class TestJolpicaEndpoints(unittest.TestCase):
    def test_get_seasons(self):
        client = JolpicaClient("http://mock")
        client._fetch_paginated_parallel = MagicMock(return_value=[
            {"SeasonTable": {"Seasons": [{"season": "2021"}, {"season": "2022"}]}},
            {"SeasonTable": {"Seasons": [{"season": "2023"}]}}
        ])
        seasons = client.get_seasons()
        self.assertEqual(len(seasons), 3)
        self.assertEqual(seasons[0]["season"], "2021")
        self.assertEqual(seasons[2]["season"], "2023")

    def test_get_season_schedule(self):
        client = JolpicaClient("http://mock")
        client._get = MagicMock(return_value={
            "MRData": {"RaceTable": {"Races": [{"round": "1", "raceName": "Bahrain"}]}}
        })
        schedule = client.get_season_schedule("2021")
        self.assertEqual(len(schedule), 1)
        self.assertEqual(schedule[0]["round"], "1")

    def test_get_event(self):
        client = JolpicaClient("http://mock")
        client.get_season_schedule = MagicMock(return_value=[
            {"round": "1", "raceName": "Bahrain"},
            {"round": "2", "raceName": "Imola"}
        ])
        event = client.get_event("2021", "2")
        self.assertIsNotNone(event)
        self.assertEqual(event["raceName"], "Imola")

        # Test not found
        event_not_found = client.get_event("2021", "3")
        self.assertIsNone(event_not_found)

        # Test exception handling
        client.get_season_schedule.side_effect = Exception("API Error")
        event_error = client.get_event("2021", "1")
        self.assertIsNone(event_error)

    def test_get_race_results(self):
        client = JolpicaClient("http://mock")
        client._get = MagicMock(return_value={
            "MRData": {"RaceTable": {"Races": [{"Results": [{"position": "1"}]}]}}
        })
        results = client.get_race_results("2021", "1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["position"], "1")

        # Test empty
        client._get = MagicMock(return_value={"MRData": {}})
        self.assertEqual(client.get_race_results("2021", "1"), [])

    def test_get_qualifying_results(self):
        client = JolpicaClient("http://mock")
        client._get = MagicMock(return_value={
            "MRData": {"RaceTable": {"Races": [{"QualifyingResults": [{"position": "1"}]}]}}
        })
        results = client.get_qualifying_results("2021", "1")
        self.assertEqual(len(results), 1)

    def test_get_sprint_results(self):
        client = JolpicaClient("http://mock")
        client._get = MagicMock(return_value={
            "MRData": {"RaceTable": {"Races": [{"SprintResults": [{"position": "1"}]}]}}
        })
        results = client.get_sprint_results("2021", "1")
        self.assertEqual(len(results), 1)

    def test_get_drivers_for_season(self):
        client = JolpicaClient("http://mock")
        client._get = MagicMock(return_value={
            "MRData": {"DriverTable": {"Drivers": [{"driverId": "hamilton"}]}}
        })
        drivers = client.get_drivers_for_season("2021")
        self.assertEqual(len(drivers), 1)

    def test_get_season_entry_list(self):
        client = JolpicaClient("http://mock")

        # Mock get_drivers_for_season
        client.get_drivers_for_season = MagicMock(return_value=[
            {"driverId": "hamilton", "code": "HAM"},
            {"driverId": "verstappen", "code": "VER"}
        ])

        # Mock _get for constructors
        def mock_get(path, **kwargs):
            if "hamilton" in path:
                return {"MRData": {"ConstructorTable": {"Constructors": [{"constructorId": "mercedes"}]}}}
            elif "verstappen" in path:
                return {"MRData": {"ConstructorTable": {"Constructors": [{"constructorId": "red_bull"}]}}}
            return {"MRData": {}}

        client._get = MagicMock(side_effect=mock_get)

        entries = client.get_season_entry_list("2021")
        self.assertEqual(len(entries), 2)

        ham_entry = next((e for e in entries if e["Driver"]["driverId"] == "hamilton"), None)
        self.assertIsNotNone(ham_entry)
        self.assertEqual(ham_entry["Constructor"]["constructorId"], "mercedes")

        # Test empty drivers
        client.get_drivers_for_season = MagicMock(return_value=[])
        self.assertEqual(client.get_season_entry_list("2021"), [])

        # Test error handling
        client.get_drivers_for_season = MagicMock(side_effect=Exception("API Error"))
        self.assertEqual(client.get_season_entry_list("2021"), [])

    def test_get_latest_season_and_round(self):
        client = JolpicaClient("http://mock")
        client._get = MagicMock(return_value={
            "MRData": {"RaceTable": {"Races": [{"season": "2021", "round": "1"}]}}
        })
        season, rnd = client.get_latest_season_and_round()
        self.assertEqual(season, "2021")
        self.assertEqual(rnd, "1")

    def test_get_next_round(self):
        client = JolpicaClient("http://mock")
        client._get = MagicMock(return_value={
            "MRData": {"RaceTable": {"Races": [{"season": "2021", "round": "2"}]}}
        })
        season, rnd = client.get_next_round()
        self.assertEqual(season, "2021")
        self.assertEqual(rnd, "2")

if __name__ == '__main__':
    unittest.main()
