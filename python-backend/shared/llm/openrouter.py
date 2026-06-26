"""
Shared OpenRouter chat-completion wrapper.

Both agents' classifiers POST through this: requests + tenacity retry on
transient errors only (5xx / timeout, never 4xx). Callers build their own
payload and parse/validate the returned JSON body. Lightweight by design.
"""
from __future__ import annotations

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class ServerError(Exception):
    """Raised on 5xx so tenacity retries; 4xx raises HTTPError (not retried)."""


@retry(
    retry=retry_if_exception_type((ServerError, requests.exceptions.Timeout)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def chat_completion(
    base_url: str,
    api_key: str,
    payload: dict,
    *,
    referer: str = "https://kiteai.dev",
    title: str = "KiteAI",
    timeout: int = 60,
) -> dict:
    """POST payload to {base_url}/chat/completions and return the parsed JSON body.

    Retries 5xx and timeouts (up to 4 attempts, exponential backoff). Other 4xx
    raise requests.HTTPError immediately. Raises on transport errors after
    retries — callers that prefer None-on-failure should wrap this in try/except.
    """
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": referer,
            "X-Title": title,
        },
        json=payload,
        timeout=timeout,
    )
    if 500 <= resp.status_code < 600:
        raise ServerError(f"{resp.status_code} from OpenRouter")
    resp.raise_for_status()
    return resp.json()
