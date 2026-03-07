import pandas as pd
from unittest.mock import MagicMock, patch
from f1pred.roster import _get_canonical_mapping, _roster_from_fastf1

def test_get_canonical_mapping():
    jc = MagicMock()
    jc.get_season_entry_list.return_value = [
        {
            "Driver": {
                "driverId": "max_verstappen",
                "permanentNumber": "1",
                "code": "VER",
                "givenName": "Max",
                "familyName": "Verstappen"
            },
            "Constructor": {
                "constructorId": "red_bull",
                "name": "Red Bull Racing"
            }
        },
        {
            "Driver": {
                "driverId": "lando_norris",
                "permanentNumber": "4",
                "code": "NOR",
                "givenName": "Lando",
                "familyName": "Norris"
            },
            "Constructor": {
                "constructorId": "mclaren",
                "name": "McLaren"
            }
        }
    ]

    mapping = _get_canonical_mapping(jc, "2024")

    assert "drivers" in mapping
    assert "constructors" in mapping

    # Check driver mapping by various keys
    assert mapping["drivers"]["1"] == "max_verstappen"
    assert mapping["drivers"]["VER"] == "max_verstappen"
    assert mapping["drivers"]["max verstappen"] == "max_verstappen"
    assert mapping["drivers"]["4"] == "lando_norris"
    assert mapping["drivers"]["NOR"] == "lando_norris"
    assert mapping["drivers"]["lando norris"] == "lando_norris"

    # Check constructor mapping
    assert mapping["constructors"]["red bull racing"] == "red_bull"
    assert mapping["constructors"]["mclaren"] == "mclaren"

@patch("f1pred.data.fastf1_backend.get_event")
def test_roster_from_fastf1_with_mapping(mock_get_event):
    # Setup mock FastF1 results
    mock_results = pd.DataFrame([
        {
            "Abbreviation": "VER",
            "DriverNumber": 1,
            "FirstName": "Max",
            "LastName": "Verstappen",
            "TeamName": "Red Bull Racing"
        },
        {
            "Abbreviation": "NOR",
            "DriverNumber": 4,
            "FirstName": "Lando",
            "LastName": "Norris",
            "TeamName": "McLaren"
        },
        {
            "Abbreviation": "BEA",
            "DriverNumber": 38,
            "FirstName": "Oliver",
            "LastName": "Bearman",
            "TeamName": "Ferrari"
        }
    ])

    mock_sess = MagicMock()
    mock_sess.results = mock_results

    mock_event = MagicMock()
    mock_event.get_session.return_value = mock_sess
    mock_get_event.return_value = mock_event

    # Canonical mapping
    mapping = {
        "drivers": {
            "VER": "max_verstappen",
            "NOR": "lando_norris",
            "38": "bearman"
        },
        "constructors": {
            "red bull racing": "red_bull",
            "mclaren": "mclaren",
            "ferrari": "ferrari"
        }
    }

    roster = _roster_from_fastf1(2024, 1, mapping=mapping)

    assert len(roster) == 3

    # Check canonicalized IDs
    ver = next(d for d in roster if d["code"] == "VER")
    assert ver["driverId"] == "max_verstappen"
    assert ver["constructorId"] == "red_bull"

    nor = next(d for d in roster if d["code"] == "NOR")
    assert nor["driverId"] == "lando_norris"
    assert nor["constructorId"] == "mclaren"

    bea = next(d for d in roster if d["code"] == "BEA")
    assert bea["driverId"] == "bearman" # Mapped via number "38"
    assert bea["constructorId"] == "ferrari"

@patch("f1pred.data.fastf1_backend.get_event")
def test_roster_from_fastf1_fallback(mock_get_event):
    # Setup mock FastF1 results with no mapping
    mock_results = pd.DataFrame([
        {
            "Abbreviation": "VER",
            "DriverNumber": 1,
            "FirstName": "Max",
            "LastName": "Verstappen",
            "TeamName": "Red Bull Racing"
        }
    ])

    mock_sess = MagicMock()
    mock_sess.results = mock_results

    mock_event = MagicMock()
    mock_event.get_session.return_value = mock_sess
    mock_get_event.return_value = mock_event

    roster = _roster_from_fastf1(2024, 1, mapping=None)

    assert len(roster) == 1
    assert roster[0]["driverId"] == "ver" # Fallback to lowercase abbreviation
    assert roster[0]["constructorId"] == "red_bull_racing" # Fallback to normalized TeamName
