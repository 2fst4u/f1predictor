import logging
import io
import re
import pytest
from f1pred.util import SafeLogFormatter, configure_logging, get_logger

def test_safe_log_formatter_sanitizes_newlines():
    """Verify that SafeLogFormatter replaces newlines with spaces."""
    # Setup
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    formatter = SafeLogFormatter(fmt="%(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger("test_newlines")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Test
    logger.info("Line 1\nLine 2")

    # Assert
    output = stream.getvalue()
    assert "Line 1 Line 2" in output
    assert "\n" not in output.strip()  # Only the final newline from logging is acceptable?
    # StreamHandler adds a newline at the end of the emission? Default implementation of Handler.emit calls stream.write(msg + terminator)
    # terminator is \n by default. But formatter output shouldn't have internal newlines.

    # Check that the internal newline is gone
    assert "Line 1\nLine 2" not in output

def test_safe_log_formatter_strips_ansi():
    """Verify that SafeLogFormatter removes ANSI codes."""
    # Setup
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    formatter = SafeLogFormatter(fmt="%(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger("test_ansi")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Test
    logger.info("\033[31mRed Alert\033[0m")

    # Assert
    output = stream.getvalue()
    assert "Red Alert" in output
    assert "\033" not in output


