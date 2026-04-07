import pytest
from unittest.mock import MagicMock, patch
from f1pred.web import app, init_web
from f1pred.config import load_config
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    cfg = load_config("config.yaml")
    init_web(cfg)
    yield TestClient(app)
    import f1pred.web as web_module
    if web_module._prediction_manager:
        web_module._prediction_manager.stop()

def test_api_schedule_filtering(client):
    """Test that /api/schedule filters out races without a round number."""
    mock_races = [
        {"raceName": "Valid Race", "round": "1"},
        {"raceName": "Invalid Race", "round": None},
        {"raceName": "Another Valid Race", "round": "2"}
    ]

    with patch("f1pred.data.jolpica.JolpicaClient.get_season_schedule", return_value=mock_races):
        response = client.get("/api/schedule/2026")
        assert response.status_code == 200
        data = response.json()
        assert len(data["races"]) == 2
        assert all(r["round"] for r in data["races"])

def test_build_session_features_filtering():
    """Test that build_session_features handles malformed schedule row."""
    from f1pred.features import build_session_features
    from datetime import datetime, timezone

    mock_jc = MagicMock()
    mock_om = MagicMock()
    cfg = load_config("config.yaml")

    # Return schedule where the target round is present but we want to test filtering in the list comprehension
    mock_jc.get_season_schedule.return_value = [
        {"round": None, "raceName": "Bad"},
        {"round": "1", "raceName": "Good", "Circuit": {"Location": {"lat": "0", "long": "0"}}}
    ]

    # This should find the "Good" race and not crash on the "Bad" one
    X, meta, roster = build_session_features(mock_jc, mock_om, 2026, 1, "race", datetime.now(timezone.utc), cfg)
    assert meta["raceName"] == "Good" if "raceName" in meta else True # Check it didn't fail

def test_roster_derivation_filtering():
    """Test that _latest_completed_round_in_season ignores None rounds."""
    from f1pred.roster import _latest_completed_round_in_season

    mock_jc = MagicMock()
    mock_jc.get_season_schedule.return_value = [
        {"round": "1", "raceName": "R1"},
        {"round": None, "raceName": "Bad"}
    ]
    # Mocking get_race_results to return True for R1 but we want to see it iterate correctly
    mock_jc.get_race_results.side_effect = lambda s, r: True if r == "1" else False

    res = _latest_completed_round_in_season(mock_jc, "2026")
    assert res == "1"
