"""Unit tests for f1pred.auth.

auth.py was effectively untested: password verification, the deliberate
72-byte bcrypt truncation, JWT creation/expiry, and the get_current_user
401 branches (invalid token, missing ``sub``, unknown user) had no coverage.
These tests exercise the primitives directly against an in-memory SQLite DB.
"""
import asyncio
from datetime import timedelta

import pytest
from jose import jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi import HTTPException

from f1pred import auth
from f1pred.database import Base
from f1pred.models_db import User


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = Session()
    db.add(User(username="alice", hashed_password=auth.get_password_hash("s3cret")))
    db.commit()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------
# Password hashing / verification
# --------------------------------------------------------------------------

class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        h = auth.get_password_hash("hunter2")
        assert h != "hunter2"
        assert h.startswith("$2")  # bcrypt prefix

    def test_correct_password_verifies(self):
        h = auth.get_password_hash("correct horse")
        assert auth.verify_password("correct horse", h) is True

    def test_wrong_password_rejected(self):
        h = auth.get_password_hash("correct horse")
        assert auth.verify_password("battery staple", h) is False

    def test_passwords_truncated_at_72_bytes(self):
        """Bcrypt only considers the first 72 bytes; auth.py truncates to match.

        Two passwords sharing a 72-byte prefix must verify interchangeably
        rather than raising on the >72-byte input.
        """
        prefix = "a" * 72
        h = auth.get_password_hash(prefix + "EXTRA-IGNORED")
        assert auth.verify_password(prefix + "different-tail", h) is True

    def test_differing_within_72_bytes_rejected(self):
        h = auth.get_password_hash("a" * 71 + "X")
        assert auth.verify_password("a" * 71 + "Y", h) is False


# --------------------------------------------------------------------------
# JWT creation
# --------------------------------------------------------------------------

class TestAccessToken:
    def test_token_round_trips_subject(self):
        token = auth.create_access_token({"sub": "alice"})
        payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        assert payload["sub"] == "alice"
        assert "exp" in payload

    def test_custom_expiry_is_honoured(self):
        short = auth.create_access_token({"sub": "alice"}, expires_delta=timedelta(seconds=1))
        long = auth.create_access_token({"sub": "alice"}, expires_delta=timedelta(hours=1))
        exp_short = jwt.decode(short, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])["exp"]
        exp_long = jwt.decode(long, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])["exp"]
        assert exp_long > exp_short


# --------------------------------------------------------------------------
# get_current_user dependency
# --------------------------------------------------------------------------

def _resolve_user(db, token):
    get_current_user = auth.get_current_user_dependency(lambda: db)
    return asyncio.run(get_current_user(token=token, db=db))


class TestGetCurrentUser:
    def test_valid_token_returns_user(self, db_session):
        token = auth.create_access_token({"sub": "alice"})
        user = _resolve_user(db_session, token)
        assert user.username == "alice"

    def test_invalid_token_raises_401(self, db_session):
        with pytest.raises(HTTPException) as exc:
            _resolve_user(db_session, "not.a.jwt")
        assert exc.value.status_code == 401

    def test_expired_token_raises_401(self, db_session):
        token = auth.create_access_token({"sub": "alice"}, expires_delta=timedelta(seconds=-1))
        with pytest.raises(HTTPException) as exc:
            _resolve_user(db_session, token)
        assert exc.value.status_code == 401

    def test_missing_subject_raises_401(self, db_session):
        token = auth.create_access_token({"sub": None})
        with pytest.raises(HTTPException) as exc:
            _resolve_user(db_session, token)
        assert exc.value.status_code == 401

    def test_unknown_user_raises_401(self, db_session):
        token = auth.create_access_token({"sub": "ghost"})
        with pytest.raises(HTTPException) as exc:
            _resolve_user(db_session, token)
        assert exc.value.status_code == 401

    def test_token_signed_with_wrong_key_raises_401(self, db_session):
        forged = jwt.encode({"sub": "alice"}, "attacker-key", algorithm=auth.ALGORITHM)
        with pytest.raises(HTTPException) as exc:
            _resolve_user(db_session, forged)
        assert exc.value.status_code == 401
