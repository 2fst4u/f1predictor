import pytest
from fastapi.testclient import TestClient
from f1pred.web import app, init_web
from f1pred.config import load_config
import os

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
