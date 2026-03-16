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
    return TestClient(app)

def test_read_main(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "F1 OUTCOME PREDICTOR" in response.text

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
