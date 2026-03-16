import pytest
from fastapi.testclient import TestClient

from f1pred.web import app, init_web
from f1pred.config import load_config

@pytest.fixture
def client():
    # Use real config for simple init
    cfg = load_config("config.yaml")
    init_web(cfg)
    return TestClient(app)

def test_get_predict_too_many_sessions(client):
    # Pass 11 session parameters
    sessions = ["race"] * 11
    url = "/api/predict?" + "&".join([f"sessions={s}" for s in sessions])
    response = client.get(url)
    assert response.status_code == 400
    assert response.json() == {"detail": "Too many sessions requested"}

def test_get_predict_stream_too_many_sessions(client):
    # Pass 11 session parameters
    sessions = ["race"] * 11
    url = "/api/predict/stream?" + "&".join([f"sessions={s}" for s in sessions])
    response = client.get(url)
    assert response.status_code == 400
    assert response.json() == {"detail": "Too many sessions requested"}
