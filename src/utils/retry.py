"""Retry utilities with exponential backoff."""

import time
from functools import wraps
from typing import Callable, Tuple, Type

import structlog

log = structlog.stdlib.get_logger()


def exponential_backoff_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        log.error(
                            "max_retries_reached",
                            function=func.__name__,
                            max_retries=max_retries,
                            error=str(e),
                        )
                        raise

                    # Calculate exponential backoff delay
                    delay = min(base_delay * (2**attempt), max_delay)

                    log.warning(
                        "retrying_after_error",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay_seconds=delay,
                        error=str(e),
                    )

                    time.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator
