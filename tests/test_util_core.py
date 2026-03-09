import logging
from f1pred.util import configure_logging, session_with_retries, __version__

def test_configure_logging():
    # Arrange
    logger = logging.getLogger()
    original_level = logger.level

    try:
        # Act
        configure_logging("DEBUG")

        # Assert
        assert logger.level == logging.DEBUG

        # Verify the formatter is SafeLogFormatter
        handlers = logger.handlers
        assert len(handlers) >= 1
        formatter = handlers[0].formatter
        assert formatter.__class__.__name__ == "SafeLogFormatter"

        # Act again to test fallback level (invalid level string)
        configure_logging("INVALID_LEVEL")
        assert logger.level == logging.WARNING

    finally:
        # Clean up
        logger.setLevel(original_level)

def test_session_with_retries():
    # Arrange & Act
    session = session_with_retries(total=2, connect=1, read=1)

    # Assert
    assert session.headers["User-Agent"].startswith("f1predictor/")
    # Ensure it's not a generic placeholder
    assert session.headers["User-Agent"] != "f1predictor/0.0.0"
    assert session.headers["User-Agent"] == f"f1predictor/{__version__}"

    # Verify the HTTPAdapter is mounted and has correct max_retries
    adapter = session.get_adapter("http://")
    assert adapter.max_retries.total == 2
    assert adapter.max_retries.connect == 1
    assert adapter.max_retries.read == 1

def test_init_caches_disabled():
    # Arrange
    from f1pred.util import init_caches
    class MockConfig:
        pass
    cfg = MockConfig()

    # Act
    # Calling init_caches with disable_cache=True should return immediately.
    # We verify it doesn't raise any errors or try to access missing config attributes.
    init_caches(cfg, disable_cache=True)

    # Assert
    # If no exception is raised, the test passes and coverage is achieved for lines 133-134.
    assert True
