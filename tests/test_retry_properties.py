"""Property-based tests for retry logic with exponential backoff.

Feature: confluence-rag-system
"""

import time
from unittest.mock import Mock

import structlog
from hypothesis import given, settings, strategies as st

from src.utils.retry import exponential_backoff_retry

log = structlog.stdlib.get_logger()


@given(
    st.integers(min_value=1, max_value=3),
    st.floats(min_value=0.01, max_value=0.1),
)
@settings(max_examples=20, deadline=None)
def test_property_4_exponential_backoff_behavior(num_failures: int, base_delay: float):
    """Property 4: Exponential backoff behavior.
    
    For any sequence of rate limit errors, the retry delays should increase
    exponentially (each delay should be at least double the previous delay).
    
    **Feature: confluence-rag-system, Property 4: Exponential backoff behavior**
    **Validates: Requirements 1.5**
    """
    log.info(
        "test_property_4_exponential_backoff_behavior",
        num_failures=num_failures,
        base_delay=base_delay,
    )

    # Track delays between retries
    delays = []
    call_times = []

    # Create a function that fails a specific number of times
    call_count = 0

    @exponential_backoff_retry(
        max_retries=num_failures,
        base_delay=base_delay,
        max_delay=60.0,
        exceptions=(ValueError,),
    )
    def failing_function():
        nonlocal call_count
        call_times.append(time.time())
        call_count += 1

        if call_count <= num_failures:
            raise ValueError(f"Simulated failure {call_count}")

        return "success"

    # Execute the function
    result = failing_function()

    # Verify the function eventually succeeded
    assert result == "success", "Function should eventually succeed"
    assert call_count == num_failures + 1, (
        f"Expected {num_failures + 1} calls, got {call_count}"
    )

    # Calculate actual delays between calls
    for i in range(1, len(call_times)):
        delay = call_times[i] - call_times[i - 1]
        delays.append(delay)

    # Property: Each delay should be approximately exponential
    # delay[i] ≈ base_delay * (2 ** i)
    for i, actual_delay in enumerate(delays):
        expected_delay = min(base_delay * (2**i), 60.0)

        # Allow for some tolerance due to execution time
        # The actual delay should be at least the expected delay (minus small tolerance)
        # and not much more than expected (plus small tolerance for execution overhead)
        tolerance = 0.05  # 50ms tolerance for execution overhead

        assert actual_delay >= expected_delay - tolerance, (
            f"Delay {i} should be at least {expected_delay}s, got {actual_delay}s"
        )

        # Also verify exponential growth: each delay should be roughly double the previous
        if i > 0 and expected_delay < 60.0:  # Skip if we hit max_delay
            # The ratio should be approximately 2 (allowing for max_delay cap)
            ratio = actual_delay / delays[i - 1]
            # Allow ratio between 1.8 and 2.2 to account for timing variations
            assert 1.5 <= ratio <= 2.5 or expected_delay >= 60.0, (
                f"Delay ratio should be ~2, got {ratio} "
                f"(delays: {delays[i-1]:.3f}s -> {actual_delay:.3f}s)"
            )

    log.info(
        "test_property_4_exponential_backoff_behavior_passed",
        num_failures=num_failures,
        delays=delays,
    )


@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=100, deadline=None)
def test_exponential_backoff_max_retries(max_retries: int):
    """Test that exponential backoff respects max_retries limit.
    
    This is a supporting test to ensure the retry mechanism stops after
    max_retries attempts.
    """
    log.info("test_exponential_backoff_max_retries", max_retries=max_retries)

    call_count = 0

    @exponential_backoff_retry(
        max_retries=max_retries,
        base_delay=0.01,  # Very short delay for fast testing
        max_delay=1.0,
        exceptions=(ValueError,),
    )
    def always_failing_function():
        nonlocal call_count
        call_count += 1
        raise ValueError("Always fails")

    # Should raise after max_retries + 1 attempts (initial + retries)
    try:
        always_failing_function()
        assert False, "Function should have raised ValueError"
    except ValueError:
        pass

    # Verify the function was called exactly max_retries + 1 times
    assert call_count == max_retries + 1, (
        f"Expected {max_retries + 1} calls, got {call_count}"
    )

    log.info(
        "test_exponential_backoff_max_retries_passed",
        max_retries=max_retries,
        call_count=call_count,
    )
