import pytest
from fastapi.testclient import TestClient

# NOTE: reference the live module (f1pred.web) at runtime rather than binding
# `app`/`init_web` at import time. Another test module (test_config_security)
# calls importlib.reload(f1pred.web), which rebinds the module's `app` and its
# `_login_attempts` counter. Referencing the live module guarantees the
# TestClient's app and the counter we clear come from the *same* module object,
# keeping the rate-limit test deterministic regardless of suite ordering.
import f1pred.web as fweb
from f1pred.config import load_config


@pytest.fixture(autouse=True)
def reset_login_attempts():
    """Clear the module-global login-attempt counter before/after each test.

    Without this, _login_attempts persists across tests in the same process,
    making the rate-limit test order-dependent.
    """
    fweb._login_attempts.clear()
    yield
    fweb._login_attempts.clear()


@pytest.fixture
def client():
    # Use real config for simple init
    cfg = load_config("config.yaml")
    fweb.init_web(cfg)
    yield TestClient(fweb.app)
    if fweb._prediction_manager:
        fweb._prediction_manager.stop()


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
    # Login to get token
    response = client.post("/api/auth/token", data={"username": "admin", "password": "admin"})
    token = response.json()["access_token"]

    client.headers.update({"Authorization": f"Bearer {token}"})

    yield client
    if fweb._prediction_manager:
        fweb._prediction_manager.stop()


def test_webhook_ssrf_protection_invalid_scheme(client_with_auth):
    response = client_with_auth.post("/api/settings/test-webhook", json={"url": "file:///etc/passwd"})
    assert response.status_code == 400


def test_webhook_ssrf_protection_non_discord(client_with_auth):
    response = client_with_auth.post("/api/settings/test-webhook", json={"url": "https://example.com"})
    assert response.status_code == 400


def test_webhook_ssrf_protection_startswith_bypass(client_with_auth):
    response = client_with_auth.post("/api/settings/test-webhook", json={"url": "https://discord.com@127.0.0.1:80/api/webhooks/"})
    assert response.status_code == 400


def test_webhook_test_success(client_with_auth, monkeypatch):
    """A valid Discord webhook URL with a successful POST returns success.

    The outbound Discord call is stubbed so the test stays offline and
    deterministic while still exercising the embed-building success path.
    """
    captured = {}

    class _FakeResp:
        status_code = 204
        text = ""

    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, timeout=None):
            captured["url"] = url
            captured["payload"] = json
            return _FakeResp()

    monkeypatch.setattr(fweb.httpx, "AsyncClient", lambda *a, **k: _FakeAsyncClient())

    response = client_with_auth.post(
        "/api/settings/test-webhook",
        json={"url": "https://discord.com/api/webhooks/123/abc"},
    )
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    # The success path builds and sends a single embed to the given URL.
    assert captured["url"] == "https://discord.com/api/webhooks/123/abc"
    assert "embeds" in captured["payload"]
    assert captured["payload"]["embeds"][0]["title"] == "✅ Webhook Connected"


def test_webhook_test_discord_error(client_with_auth, monkeypatch):
    """A Discord API error (4xx/5xx) is surfaced as a failure to the client."""
    class _FakeResp:
        status_code = 400
        text = "Bad Request"

    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, timeout=None):
            return _FakeResp()

    monkeypatch.setattr(fweb.httpx, "AsyncClient", lambda *a, **k: _FakeAsyncClient())

    response = client_with_auth.post(
        "/api/settings/test-webhook",
        json={"url": "https://discord.com/api/webhooks/123/abc"},
    )
    # The endpoint surfaces the upstream failure as an error response (the
    # inner HTTPException is remapped to 500 by the broad exception handler).
    assert response.status_code >= 400


def test_login_rate_limiting(client):
    # Reset immediately before exercising the limit so the test is independent
    # of any login attempts made by earlier tests/fixtures.
    fweb._login_attempts.clear()

    # With a freshly-cleared counter, the first 10 failed attempts are rejected
    # with 401 (bad credentials), not throttled.
    for _ in range(10):
        response = client.post("/api/auth/token", data={"username": "admin", "password": "wrongpassword"})
        assert response.status_code == 401

    # The 11th attempt within the window is rate limited.
    response = client.post("/api/auth/token", data={"username": "admin", "password": "wrongpassword"})
    assert response.status_code == 429
    assert "Too many login attempts" in response.json()["detail"]
