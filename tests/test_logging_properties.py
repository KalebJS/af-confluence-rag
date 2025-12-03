"""Property-based tests for logging functionality.

**Feature: confluence-rag-system, Property 31: Error log format**

This module tests that error logs contain required fields:
- timestamp
- severity level (log level)
- error message

**Validates: Requirements 9.1**
"""

import json
import logging
from datetime import datetime
from io import StringIO

import structlog
from hypothesis import given, settings
from hypothesis import strategies as st

from src.utils.logging_config import configure_logging


@given(
    log_level=st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    error_message=st.text(min_size=1, max_size=200),
)
@settings(max_examples=100)
def test_error_log_format_contains_required_fields(
    log_level: str,
    error_message: str,
) -> None:
    """
    Property 31: Error log format

    *For any* logged error, the log entry should contain timestamp,
    severity level, and error message fields.

    **Validates: Requirements 9.1**

    This property ensures that all error logs (and logs in general) contain
    the required fields for proper monitoring and debugging.

    Args:
        log_level: The log level to test (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        error_message: The error message to log
    """
    # Create a string buffer to capture log output
    log_buffer = StringIO()

    # Configure logging to output JSON to our buffer
    # We need to reconfigure structlog for each test to use our buffer
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG,
        stream=log_buffer,
        force=True,  # Force reconfiguration
    )

    # Configure structlog with JSON output
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,  # Don't cache for testing
    )

    # Get a logger and log a message at the specified level
    log = structlog.stdlib.get_logger("test_logger")

    # Log the message at the appropriate level
    log_method = getattr(log, log_level.lower())
    log_method("test_event", error=error_message)

    # Get the logged output
    log_output = log_buffer.getvalue().strip()

    # Parse the JSON log entry
    try:
        log_entry = json.loads(log_output)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Log output is not valid JSON: {log_output}") from e

    # Property: Log entry must contain timestamp field
    assert "timestamp" in log_entry, (
        f"Log entry missing 'timestamp' field. "
        f"Log entry: {log_entry}"
    )

    # Verify timestamp is a valid ISO format datetime
    try:
        datetime.fromisoformat(log_entry["timestamp"].replace("Z", "+00:00"))
    except (ValueError, AttributeError) as e:
        raise AssertionError(
            f"Timestamp is not in valid ISO format: {log_entry['timestamp']}"
        ) from e

    # Property: Log entry must contain log level (severity) field
    assert "level" in log_entry, (
        f"Log entry missing 'level' field. "
        f"Log entry: {log_entry}"
    )

    # Verify log level matches what we logged
    assert log_entry["level"].upper() == log_level.upper(), (
        f"Log level mismatch. Expected: {log_level}, Got: {log_entry['level']}"
    )

    # Property: Log entry must contain the error message
    # The message could be in 'event' or 'error' field depending on how it's logged
    assert "event" in log_entry or "error" in log_entry, (
        f"Log entry missing message field ('event' or 'error'). "
        f"Log entry: {log_entry}"
    )

    # Verify the error message is present
    if "error" in log_entry:
        assert log_entry["error"] == error_message, (
            f"Error message mismatch. Expected: {error_message}, "
            f"Got: {log_entry['error']}"
        )


@given(
    error_message=st.text(min_size=1, max_size=200),
    context_key=st.text(
        min_size=1,
        max_size=20,
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=97),
    ),
    context_value=st.one_of(
        st.text(min_size=0, max_size=100),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
    ),
)
@settings(max_examples=100)
def test_error_log_format_preserves_context(
    error_message: str,
    context_key: str,
    context_value: str | int | float | bool,
) -> None:
    """
    Property 31 (Extended): Error log format with context

    *For any* logged error with context, the log entry should contain
    timestamp, severity level, error message, AND the context fields.

    **Validates: Requirements 9.1**

    This property ensures that contextual information (like page_id, space_key, etc.)
    is preserved in log entries for debugging.

    Args:
        error_message: The error message to log
        context_key: A context field name
        context_value: A context field value
    """
    # Create a string buffer to capture log output
    log_buffer = StringIO()

    # Configure logging to output JSON to our buffer
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG,
        stream=log_buffer,
        force=True,
    )

    # Configure structlog with JSON output
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    # Get a logger and log an error with context
    log = structlog.stdlib.get_logger("test_logger")

    # Log with context using kwargs
    log.error("test_event", error=error_message, **{context_key: context_value})

    # Get the logged output
    log_output = log_buffer.getvalue().strip()

    # Parse the JSON log entry
    try:
        log_entry = json.loads(log_output)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Log output is not valid JSON: {log_output}") from e

    # Property: Log entry must contain all required fields
    assert "timestamp" in log_entry, "Log entry missing 'timestamp' field"
    assert "level" in log_entry, "Log entry missing 'level' field"
    assert log_entry["level"].upper() == "ERROR", "Log level should be ERROR"

    # Property: Log entry must preserve context
    assert context_key in log_entry, (
        f"Log entry missing context field '{context_key}'. "
        f"Log entry: {log_entry}"
    )

    # Verify context value matches (with type conversion for JSON)
    logged_value = log_entry[context_key]

    # JSON converts all numbers to float/int, so we need to handle that
    if isinstance(context_value, bool):
        assert logged_value == context_value, (
            f"Context value mismatch for '{context_key}'. "
            f"Expected: {context_value}, Got: {logged_value}"
        )
    elif isinstance(context_value, (int, float)):
        # Allow for floating point comparison
        assert abs(float(logged_value) - float(context_value)) < 1e-6, (
            f"Context value mismatch for '{context_key}'. "
            f"Expected: {context_value}, Got: {logged_value}"
        )
    else:
        assert logged_value == context_value, (
            f"Context value mismatch for '{context_key}'. "
            f"Expected: {context_value}, Got: {logged_value}"
        )


def test_logging_configuration_creates_valid_json_logs() -> None:
    """
    Test that the logging configuration creates valid JSON logs.

    This is a simple example-based test to verify the logging configuration
    works correctly.
    """
    # Create a string buffer to capture log output
    log_buffer = StringIO()

    # Configure logging
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        stream=log_buffer,
        force=True,
    )

    # Use our configure_logging function
    configure_logging(log_level="INFO", json_logs=True, log_file=None)

    # Get a logger and log a message
    log = structlog.stdlib.get_logger("test_logger")
    log.info("test_event", test_key="test_value", count=42)

    # Get the logged output
    log_output = log_buffer.getvalue().strip()

    # Should be valid JSON
    log_entry = json.loads(log_output)

    # Should contain required fields
    assert "timestamp" in log_entry
    assert "level" in log_entry
    assert "event" in log_entry
    assert log_entry["event"] == "test_event"
    assert log_entry["test_key"] == "test_value"
    assert log_entry["count"] == 42
