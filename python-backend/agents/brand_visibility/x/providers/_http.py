from __future__ import annotations

import logging
import time

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    pass


class ServerError(Exception):
    pass


@retry(
    retry=retry_if_exception_type((RateLimitError, ServerError, requests.exceptions.Timeout)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=5, max=900),
    reraise=True,
)
def safe_get(
    session: requests.Session,
    url: str,
    params: dict,
    timeout: int = 30,
) -> requests.Response:
    resp = session.get(url, params=params, timeout=timeout)

    if resp.status_code == 429:
        reset = resp.headers.get("Retry-After") or resp.headers.get("x-ratelimit-requests-reset")
        if reset:
            try:
                sleep_for = max(5, int(reset) - int(time.time()) + 5)
                logger.warning("Rate limited. Sleeping %d s.", min(sleep_for, 900))
                time.sleep(min(sleep_for, 900))
            except (TypeError, ValueError):
                pass
        raise RateLimitError(f"429 from {url}")

    if 500 <= resp.status_code < 600:
        raise ServerError(f"{resp.status_code} from {url}")

    resp.raise_for_status()
    return resp
