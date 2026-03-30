"""
Retry utilities for handling transient API failures.

When interacting with external APIs (LLM providers, vector databases,
web services), transient failures are common due to rate limiting,
network issues, or temporary server problems. This module provides a
retry decorator that automatically retries failed operations with
exponential backoff.

Features:
    - Configurable maximum retry count.
    - Exponential backoff with jitter to prevent thundering herd.
    - Customizable exception types to retry on.
    - Optional callback for logging retry attempts.

Example:
    >>> from agentic_ai.utils.retry import retry_with_backoff
    >>> @retry_with_backoff(max_retries=3, base_delay=1.0)
    ... def call_api():
    ...     response = requests.get("https://api.example.com/data")
    ...     response.raise_for_status()
    ...     return response.json()
"""

import functools
import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

# Configure module-level logger.
logger = logging.getLogger(__name__)

# TypeVar for preserving function return types in the decorator.
F = TypeVar("F", bound=Callable[..., Any])


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator that retries a function with exponential backoff on failure.

    Wraps a function so that if it raises one of the specified exception
    types, it is automatically retried after a delay. The delay increases
    exponentially with each retry and includes optional random jitter.

    The backoff formula is:
        delay = min(base_delay * (exponential_base ** attempt), max_delay)
        if jitter: delay = delay * random(0.5, 1.5)

    Args:
        max_retries:     Maximum number of retry attempts. Default is 3.
        base_delay:      Initial delay in seconds before the first retry.
                         Default is 1.0 second.
        max_delay:       Maximum delay cap in seconds. Default is 60 seconds.
                         Prevents excessively long waits.
        exponential_base: The base for exponential growth. Default is 2.0,
                          meaning delays double with each retry.
        jitter:          Whether to add random jitter to delays. Default is True.
                         Jitter prevents multiple clients from retrying
                         simultaneously (thundering herd problem).
        retry_on:        Tuple of exception types to retry on. Default is
                         (Exception,) which retries on any exception.
                         Narrow this to specific exceptions for safety.

    Returns:
        A decorator function that wraps the target function with retry logic.

    Example:
        >>> @retry_with_backoff(max_retries=3, retry_on=(ConnectionError, TimeoutError))
        ... def fetch_data():
        ...     return requests.get("https://api.example.com").json()
    """

    def decorator(func: F) -> F:
        """
        The actual decorator that wraps the function with retry logic.

        Args:
            func: The function to wrap.

        Returns:
            The wrapped function with retry behavior.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            Execute the function with retry logic on failure.

            Attempts to call the wrapped function up to (max_retries + 1) times.
            On each failure, logs the error, calculates the backoff delay, and
            sleeps before retrying. If all retries are exhausted, the last
            exception is re-raised.
            """
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    # Attempt to call the wrapped function.
                    return func(*args, **kwargs)

                except retry_on as e:
                    last_exception = e

                    # If this was the last attempt, don't sleep - just re-raise.
                    if attempt >= max_retries:
                        logger.error(
                            "Function '%s' failed after %d retries: %s",
                            func.__name__,
                            max_retries,
                            e,
                        )
                        raise

                    # Calculate the delay with exponential backoff.
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay,
                    )

                    # Add random jitter to prevent synchronized retries.
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        "Function '%s' failed (attempt %d/%d): %s. "
                        "Retrying in %.1f seconds...",
                        func.__name__,
                        attempt + 1,
                        max_retries + 1,
                        e,
                        delay,
                    )

                    # Wait before the next retry attempt.
                    time.sleep(delay)

            # This should never be reached, but raise the last exception just in case.
            if last_exception:
                raise last_exception
            return None

        return wrapper  # type: ignore[return-value]

    return decorator
