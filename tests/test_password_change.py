import pytest
from fastapi.testclient import TestClient
from f1pred.web import app, init_web
import f1pred.web as web_module
from f1pred.config import load_config
from f1pred.auth import get_password_hash

@pytest.fixture
def auth_client():
    # Use real config for simple init
    cfg = load_config("config.yaml")
    init_web(cfg)

    # Ensure admin user has known password
    with web_module._db_session_factory() as db:
        from f1pred.models_db import User
        admin = db.query(User).filter(User.username == "admin").first()
        if admin:
            admin.hashed_password = get_password_hash("admin")
            db.commit()

    client = TestClient(app)
    yield client
    if web_module._prediction_manager:
        web_module._prediction_manager.stop()

def test_password_change(auth_client):
    # 1. Login to get token
    response = auth_client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "admin"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Change password
    response = auth_client.post(
        "/api/auth/change-password",
        json={"current_password": "admin", "new_password": "newpassword"},
        headers=headers
    )
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

    # 3. Try to login with old password (should fail)
    response = auth_client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "admin"}
    )
    assert response.status_code == 401

    # 4. Login with new password (should succeed)
    response = auth_client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "newpassword"}
    )
    assert response.status_code == 200

def test_password_change_incorrect_current(auth_client):
    # 1. Login to get token
    response = auth_client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "admin"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Change password with wrong current password
    response = auth_client.post(
        "/api/auth/change-password",
        json={"current_password": "wrongpassword", "new_password": "newpassword"},
        headers=headers
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Incorrect current password"
