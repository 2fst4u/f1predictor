import pytest
from fastapi.testclient import TestClient

from f1pred.web import app, init_web
from f1pred.config import load_config

@pytest.fixture
def client():
    # Use real config for simple init
    cfg = load_config("config.yaml")
    init_web(cfg)
    yield TestClient(app)
    import f1pred.web as web_module
    if web_module._prediction_manager:
        web_module._prediction_manager.stop()

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

@pytest.fixture
def client_with_auth(client):
    import f1pred.web as web_module

    # Login to get token
    response = client.post("/api/auth/token", data={"username": "admin", "password": "admin"})
    token = response.json()["access_token"]

    client.headers.update({"Authorization": f"Bearer {token}"})

    yield client
    if web_module._prediction_manager:
        web_module._prediction_manager.stop()

def test_webhook_ssrf_protection_invalid_scheme(client_with_auth):
    response = client_with_auth.post("/api/settings/test-webhook", json={"url": "file:///etc/passwd"})
    assert response.status_code == 400

def test_webhook_ssrf_protection_non_discord(client_with_auth):
    response = client_with_auth.post("/api/settings/test-webhook", json={"url": "https://example.com"})
    assert response.status_code == 400
