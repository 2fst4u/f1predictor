import json
import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from f1pred.web import app, init_web
import f1pred.web as web_module
from f1pred.config import load_config
from unittest.mock import patch

@pytest.fixture
def client():
    # Use real config for simple init
    cfg = load_config("config.yaml")
    init_web(cfg)
    yield TestClient(app)
    if web_module._prediction_manager:
        web_module._prediction_manager.stop()


def _fake_results():
    """A minimal but realistic run_predictions_for_event return value.

    Includes a NaN and a nested dict so the JSON sanitisation path is
    exercised deterministically (no model training involved).
    """
    ranked = pd.DataFrame([
        {
            "driverId": "ver",
            "code": "VER",
            "name": "Max Verstappen",
            "constructorName": "Red Bull",
            "predicted_position": 1,
            "p_win": 0.55,
            "p_top3": 0.9,
            "p_dnf": float("nan"),  # exercises NaN sanitisation
            "shap_values": {"form_index": -1.2, "grid": float("nan")},  # nested sanitisation
        },
        {
            "driverId": "ham",
            "code": "HAM",
            "name": "Lewis Hamilton",
            "constructorName": "Mercedes",
            "predicted_position": 2,
            "p_win": 0.2,
            "p_top3": 0.7,
            "p_dnf": 0.05,
            "shap_values": {"form_index": -0.5},
        },
    ])
    return {
        "season": 2024,
        "round": 1,
        "sessions": {
            "race": {"ranked": ranked, "meta": {"weather": {"temp_mean": 22.0}}},
        },
    }

def test_read_main(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>F1 Outcome Predictor</title>" in response.text

def test_get_config(client):
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "default_sessions" in data

def test_get_config_no_config(client):
    old_config = web_module._config
    web_module._config = None
    try:
        response = client.get("/api/config")
        assert response.status_code == 200
        assert response.json() == {"error": "Config not initialized"}
    finally:
        web_module._config = old_config

@patch("f1pred.web.resolve_event")
def test_get_config_resolve_error(mock_resolve, client):
    mock_resolve.side_effect = Exception("resolve error")
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert data["next_event"]["season"] is None

@patch("f1pred.web.JolpicaClient")
def test_get_seasons(mock_jc, client):
    mock_instance = mock_jc.return_value
    mock_instance.get_seasons.return_value = [{"season": "2023"}, {"season": "2024"}]

    response = client.get("/api/seasons")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["season"] == "2024"
    assert data[1]["season"] == "2023"

@patch("f1pred.web.JolpicaClient")
def test_get_seasons_error(mock_jc, client):
    mock_instance = mock_jc.return_value
    mock_instance.get_seasons.side_effect = Exception("API Error")

    response = client.get("/api/seasons")
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}

@patch("f1pred.web.JolpicaClient")
def test_get_schedule(mock_jc, client):
    mock_instance = mock_jc.return_value
    mock_instance.get_season_schedule.return_value = [{"round": "1", "raceName": "Bahrain"}, {"raceName": "No Round"}]

    response = client.get("/api/schedule/2024")
    assert response.status_code == 200
    data = response.json()
    assert data["season"] == "2024"
    assert len(data["races"]) == 1
    assert data["races"][0]["round"] == "1"

def test_get_schedule_invalid(client):
    # Testing error handling
    response = client.get("/api/schedule/invalid")
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}

def test_get_predict_invalid_season(client):
    # Testing generic exception catch in predict endpoint
    response = client.get("/api/predict", params={"season": "invalid"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}

@patch("f1pred.web.resolve_event")
@patch("f1pred.web.JolpicaClient")
def test_get_event_status_success(mock_jc, mock_resolve, client):
    mock_instance = mock_jc.return_value
    mock_resolve.return_value = (2024, 1, {"raceName": "Bahrain GP", "Sprint": {}, "SprintQualifying": {}})

    # Mocking Jolpica methods to return some results but not others
    mock_instance.get_race_results.return_value = []
    mock_instance.get_qualifying_results.return_value = [{"position": 1}]
    mock_instance.get_sprint_results.return_value = []

    response = client.get("/api/event-status/2024/1")
    assert response.status_code == 200
    data = response.json()
    assert data["season"] == 2024
    assert data["round"] == 1
    assert data["raceName"] == "Bahrain GP"

    sessions = data["sessions"]
    assert len(sessions) == 4

    sq_status = next(s for s in sessions if s["id"] == "sprint_qualifying")
    assert not sq_status["has_results"]

    s_status = next(s for s in sessions if s["id"] == "sprint")
    assert not s_status["has_results"]

    q_status = next(s for s in sessions if s["id"] == "qualifying")
    assert q_status["has_results"]

    r_status = next(s for s in sessions if s["id"] == "race")
    assert not r_status["has_results"]

@patch("f1pred.web.resolve_event")
def test_get_event_status_error(mock_resolve, client):
    mock_resolve.side_effect = Exception("Resolve failed")

    response = client.get("/api/event-status/2024/1")
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


# --- /api/predict ---------------------------------------------------------

@patch("f1pred.web.run_predictions_for_event")
def test_get_predict_success(mock_run, client):
    mock_run.return_value = _fake_results()

    response = client.get("/api/predict", params={"round": "1"})
    assert response.status_code == 200
    data = response.json()
    assert data["season"] == 2024
    assert data["round"] == 1
    preds = data["sessions"]["race"]["predictions"]
    assert [p["code"] for p in preds] == ["VER", "HAM"]
    # NaN values are sanitised to null for JSON safety
    assert preds[0]["p_dnf"] is None
    assert preds[0]["shap_values"]["grid"] is None
    assert data["sessions"]["race"]["weather"] == {"temp_mean": 22.0}


@patch("f1pred.web.run_predictions_for_event")
def test_get_predict_no_results(mock_run, client):
    mock_run.return_value = None
    response = client.get("/api/predict", params={"round": "1"})
    assert response.status_code == 200
    assert response.json() == {"error": "No results generated"}


def test_get_predict_too_many_sessions(client):
    sessions = "&".join(f"sessions=race" for _ in range(11))
    response = client.get(f"/api/predict?round=1&{sessions}")
    assert response.status_code == 400
    assert response.json() == {"detail": "Too many sessions requested"}


def test_get_predict_not_configured(client):
    old = web_module._config
    web_module._config = None
    try:
        response = client.get("/api/predict", params={"round": "1"})
        assert response.status_code == 500
        assert response.json() == {"detail": "Application not configured"}
    finally:
        web_module._config = old


# --- /api/predict/stream --------------------------------------------------

def _parse_sse(text):
    """Extract JSON payloads from an SSE response body."""
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


@patch("f1pred.web.run_predictions_for_event")
def test_predict_stream_success(mock_run, client):
    mock_run.return_value = _fake_results()

    response = client.get("/api/predict/stream", params={"round": "1"})
    assert response.status_code == 200
    events = _parse_sse(response.text)
    result_events = [e for e in events if e.get("type") == "results"]
    assert len(result_events) == 1
    out = result_events[0]["data"]
    assert out["season"] == 2024
    assert [p["code"] for p in out["sessions"]["race"]["predictions"]] == ["VER", "HAM"]


@patch("f1pred.web.run_predictions_for_event")
def test_predict_stream_no_results(mock_run, client):
    mock_run.return_value = None
    response = client.get("/api/predict/stream", params={"round": "1"})
    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert any(e.get("type") == "error" for e in events)


@patch("f1pred.web.run_predictions_for_event")
def test_predict_stream_exception(mock_run, client):
    mock_run.side_effect = Exception("boom")
    response = client.get("/api/predict/stream", params={"round": "1"})
    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert any(e.get("type") == "error" for e in events)


def test_predict_stream_too_many_sessions(client):
    sessions = "&".join("sessions=race" for _ in range(11))
    response = client.get(f"/api/predict/stream?round=1&{sessions}")
    assert response.status_code == 400


# --- /api/predictions/latest ---------------------------------------------

def test_predictions_latest(client):
    response = client.get("/api/predictions/latest")
    assert response.status_code == 200
    data = response.json()
    # Keys are always present even before the first prediction cycle
    assert set(["results", "diffs", "last_update", "status"]).issubset(data.keys())


def test_predictions_latest_no_manager(client):
    old = web_module._prediction_manager
    web_module._prediction_manager = None
    try:
        response = client.get("/api/predictions/latest")
        assert response.status_code == 503
    finally:
        web_module._prediction_manager = old


# --- /api/predictions/live (SSE) -----------------------------------------

def test_predictions_live_initial_events(client):
    """The live stream emits the cached prediction, recent diffs and status
    immediately on connect. We drive the async generator directly and read just
    those initial events, avoiding TestClient's buffering of the (infinite)
    streaming response and the 30s update wait loop."""
    import asyncio
    from f1pred.prediction_manager import PredictionDiff, DriverMovement

    pm = web_module._prediction_manager
    pm._latest_results = {"season": 2024, "rounds": {"1": {"sessions": {}}}}
    pm._latest_diffs = [
        PredictionDiff(
            session="race",
            movements=[DriverMovement(
                driver_id="ver", driver_name="Max", code="VER", team="Red Bull",
                old_position=2, new_position=1, direction=1, reasons=["form"],
            )],
            changed_variables=["form"],
        )
    ]

    async def collect_initial():
        resp = await web_module.predictions_live()
        gen = resp.body_iterator
        events = []
        try:
            async for chunk in gen:
                if chunk.startswith("data:"):
                    events.append(json.loads(chunk[len("data: "):]))
                if len(events) >= 3:
                    break
        finally:
            await gen.aclose()
        return events

    seen = asyncio.run(collect_initial())
    types = [e.get("type") for e in seen]
    assert "prediction" in types
    assert "diff" in types
    assert "status" in types


def test_predictions_live_no_manager(client):
    old = web_module._prediction_manager
    web_module._prediction_manager = None
    try:
        response = client.get("/api/predictions/live")
        assert response.status_code == 503
    finally:
        web_module._prediction_manager = old
