"""Centralized logging configuration for the application."""

import logging
import sys
from typing import Any

import structlog


def configure_logging(
    log_level: str = "INFO",
    json_logs: bool = True,
    log_file: str | None = None,
) -> None:
    """
    Configure structured logging for the application.

    This function sets up structlog with:
    - JSON formatting for production (when json_logs=True)
    - Console formatting for development (when json_logs=False)
    - Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Optional file output with rotation

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: If True, output JSON logs. If False, use console format.
        log_file: Optional path to log file. If None, logs only to stdout.

    Example:
        >>> configure_logging(log_level="DEBUG", json_logs=False)
        >>> log = structlog.stdlib.get_logger()
        >>> log.info("application_started", version="1.0.0")
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        stream=sys.stdout,
    )

    # Configure handlers
    handlers: list[Any] = []

    # Add file handler if log_file is specified
    if log_file:
        from logging.handlers import RotatingFileHandler

        # Create rotating file handler
        # Max size: 10MB, keep 5 backup files
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(numeric_level)
        logging.root.addHandler(file_handler)

    # Configure structlog processors
    processors: list[Any] = [
        # Add log level to event dict
        structlog.stdlib.add_log_level,
        # Add logger name to event dict
        structlog.stdlib.add_logger_name,
        # Add timestamp to event dict
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Add stack info for exceptions
        structlog.processors.StackInfoRenderer(),
        # Format exceptions
        structlog.processors.format_exc_info,
        # Add call site information (file, line, function)
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ],
        ),
    ]

    # Add appropriate renderer based on json_logs setting
    if json_logs:
        # JSON renderer for production
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Console renderer for development
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Optional logger name. If None, uses the calling module's name.

    Returns:
        Configured structlog logger

    Example:
        >>> log = get_logger(__name__)
        >>> log.info("processing_started", document_id="123")
    """
    return structlog.stdlib.get_logger(name)
