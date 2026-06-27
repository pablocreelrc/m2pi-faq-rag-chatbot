"""OpenAI client setup and a small retry-with-backoff wrapper.

Credentials come only from the environment (never hard-coded). Transient API
errors are retried with exponential backoff + jitter so a single rate-limit or
network blip does not fail the run. Ported from the M1 project's llm_client.
"""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from src.config import Settings

T = TypeVar("T")

# Transient errors worth retrying. A hard 4xx (bad request, auth) is not here.
RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, APIError)


def get_client(settings: Settings | None = None) -> OpenAI:
    settings = settings or Settings.from_env()
    return OpenAI(api_key=settings.api_key)


def call_with_retry(
    func: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> T:
    """Run ``func()``, retrying transient OpenAI errors with backoff + jitter."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except RETRYABLE as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            time.sleep(random.uniform(0, delay))  # jitter avoids thundering herd
    raise RuntimeError(
        f"OpenAI call failed after {max_attempts} attempts: {last_exc}"
    ) from last_exc
