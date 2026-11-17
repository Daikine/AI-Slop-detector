"""OpenAI Python SDK helpers for OpenAI-compatible Qwen endpoints."""
from __future__ import annotations

from openai import AsyncOpenAI, OpenAI


def create_async_client(api_key: str, base_url: str, *, timeout: float = 35.0,
                        http_client=None) -> AsyncOpenAI:
    """Create an async SDK client for a Qwen/OpenAI-compatible API.

    ``http_client`` is an optional httpx.AsyncClient transport. It is useful for
    deterministic tests; ownership remains with the caller.
    """
    if not api_key:
        raise ValueError('API key is required to create an LLM client')
    if not base_url:
        raise ValueError('API base URL is required to create an LLM client')
    options = {
        'api_key': api_key,
        'base_url': base_url.rstrip('/') + '/',
        'timeout': timeout,
        # Retries are handled by the calling layer so logs/cache behavior stay
        # deterministic and SDK retries do not multiply application retries.
        'max_retries': 0,
    }
    if http_client is not None:
        options['http_client'] = http_client
    return AsyncOpenAI(**options)


def create_sync_client(api_key: str, base_url: str, *, timeout: float = 35.0,
                       http_client=None) -> OpenAI:
    """Create a sync SDK client (used by local Qwen connection checks)."""
    if not api_key:
        raise ValueError('API key is required to create an LLM client')
    if not base_url:
        raise ValueError('API base URL is required to create an LLM client')
    options = {
        'api_key': api_key,
        'base_url': base_url.rstrip('/') + '/',
        'timeout': timeout,
        'max_retries': 0,
    }
    if http_client is not None:
        options['http_client'] = http_client
    return OpenAI(**options)
