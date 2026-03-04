import pytest
from unittest.mock import MagicMock
import json
from f1pred.util import http_get_json

def test_http_get_json_valid_json():
    session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": "100"}
    mock_resp.iter_content.return_value = [b'{"key": "value"}']
    mock_resp.encoding = "utf-8"

    # Setup context manager
    mock_resp.__enter__.return_value = mock_resp
    session.get.return_value = mock_resp

    result = http_get_json(session, "http://example.com")

    assert result == {"key": "value"}
    mock_resp.raise_for_status.assert_called_once()
    session.get.assert_called_once_with("http://example.com", params=None, timeout=30, stream=True)

def test_http_get_json_content_length_exceeded():
    session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": str(10 * 1024 * 1024 + 1)}

    # Setup context manager
    mock_resp.__enter__.return_value = mock_resp
    session.get.return_value = mock_resp

    with pytest.raises(ValueError, match="Response too large"):
        http_get_json(session, "http://example.com")

def test_http_get_json_invalid_json_fallback():
    session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": "10"}
    mock_resp.iter_content.return_value = [b'not json']
    mock_resp.encoding = "utf-8"

    # Setup context manager
    mock_resp.__enter__.return_value = mock_resp
    session.get.return_value = mock_resp

    result = http_get_json(session, "http://example.com")
    assert result == "not json"

def test_http_get_json_content_length_invalid():
    session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": "invalid"}
    mock_resp.iter_content.return_value = [b'{"key": "value"}']
    mock_resp.encoding = "utf-8"

    # Setup context manager
    mock_resp.__enter__.return_value = mock_resp
    session.get.return_value = mock_resp

    result = http_get_json(session, "http://example.com")

    assert result == {"key": "value"}

def test_http_get_json_content_length_exceeded_in_body():
    session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": "10"}

    # Send a chunk larger than MAX_SIZE
    large_chunk = b'a' * (10 * 1024 * 1024 + 1)
    mock_resp.iter_content.return_value = [large_chunk]

    # Setup context manager
    mock_resp.__enter__.return_value = mock_resp
    session.get.return_value = mock_resp

    with pytest.raises(ValueError, match="Response too large"):
        http_get_json(session, "http://example.com")

def test_http_get_json_fallback_decode_error():
    session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": "10"}
    mock_resp.iter_content.return_value = [b'\xff\xfe']
    mock_resp.encoding = "utf-8"

    # Setup context manager
    mock_resp.__enter__.return_value = mock_resp
    session.get.return_value = mock_resp

    result = http_get_json(session, "http://example.com")
    # Will use replace error handler
    assert result == "\ufffd\ufffd"

def test_http_get_json_fallback_decode_error_replace_fails():
    session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": "10"}
    mock_resp.encoding = "utf-8"

    # Create a mock content bytearray that raises Exception on decode
    class BadByteArray(bytearray):
        def decode(self, *args, **kwargs):
            raise Exception("decode failed")

    mock_resp.iter_content.return_value = [BadByteArray(b'somebytes')]

    # Setup context manager
    mock_resp.__enter__.return_value = mock_resp
    session.get.return_value = mock_resp

    result = http_get_json(session, "http://example.com")
    assert result == "somebytes"
