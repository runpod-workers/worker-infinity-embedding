import httpx

from config import HttpClientConfig

_CLIENT_CONFIG = HttpClientConfig()


def create_http_client() -> httpx.AsyncClient:
    """
    Create a configured httpx.AsyncClient.
    
    Returns:
        httpx.AsyncClient with proper timeout, limits, and headers configured.
    """
    limits = httpx.Limits(
        max_connections=_CLIENT_CONFIG.max_connections,
        max_keepalive_connections=_CLIENT_CONFIG.max_keepalive_connections,
    )

    timeout = httpx.Timeout(_CLIENT_CONFIG.timeout_seconds)

    headers = {
        "User-Agent": _CLIENT_CONFIG.user_agent,
    }

    return httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        headers=headers,
        follow_redirects=True,
        trust_env=True,
    )
