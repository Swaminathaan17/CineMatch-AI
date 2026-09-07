"""
Thin wrapper around the TMDB API. Every call goes through here so caching,
retries, and error handling live in one place instead of being scattered
across routers.

Behavior:
- Shared HTTP client (one httpx.AsyncClient reused across requests instead of
  a fresh one per call).
- Successful GET responses are cached with a TTL so repeat reads of the same
  path+params don't hit the network again.
- Retries (with exponential backoff) for 429/500/502/503/504 plus timeouts and
  transport/network errors. Retry-After is honored for 429s. 400 and 404 are
  NOT retried.
- 404 responses are raised as TMDBNotFoundError so callers can distinguish
  "resource doesn't exist" from genuine service failures (TMDBError).
"""
from __future__ import annotations

import asyncio
import time

import httpx

from app.config import settings

RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)


class TMDBError(Exception):
    """Generic TMDB service or transport error (anything except a 404)."""


class TMDBNotFoundError(TMDBError):
    """The requested TMDB resource does not exist (HTTP 404)."""


class TMDBClient:
    def __init__(
        self,
        max_retries: int = 3,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 6.0,
        timeout: float = 10.0,
        cache_ttl: float = 300.0,
    ):
        self.base_url = settings.tmdb_base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.timeout = timeout
        self.cache_ttl = cache_ttl

        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._cache: dict[tuple, tuple[float, dict]] = {}

    @property
    def api_key(self) -> str:
        # read dynamically rather than capturing once at construction time -
        # this module is imported (and the singleton below created) at
        # startup, so a static self.api_key would silently ignore any key
        # set after that point, which also made this class hard to test
        return settings.tmdb_api_key

    @property
    def cache_size(self) -> int:
        """Number of entries currently held in the TTL cache (for tests/metrics)."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Drop all cached responses. Useful after upstream data changes."""
        self._cache.clear()

    def _params(self, extra: dict | None = None) -> dict:
        params = {"api_key": self.api_key}
        if extra:
            params.update(extra)
        return params

    def _cache_key(self, path: str, params: dict) -> tuple:
        key_params = tuple(sorted((k, str(v)) for k, v in params.items()))
        return (path, key_params)

    async def _get_client(self) -> httpx.AsyncClient:
        # One shared client. Tests spin up separate event loops (asyncio.run,
        # each TestClient portal), so if the running loop differs from the one
        # the cached client is bound to, replace it with a fresh client for the
        # current loop. In production there is a single loop, so a single
        # client is created once and reused for every request.
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._client_loop = loop
        return self._client

    def _retry_delay(self, attempt: int, resp: httpx.Response | None) -> float:
        """Exponential backoff, honoring Retry-After for 429 responses."""
        if resp is not None and resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(max(float(retry_after), 0.0), self.retry_max_delay)
                except ValueError:
                    pass
        exp = self.retry_base_delay * (2 ** (attempt - 1))
        return min(exp, self.retry_max_delay)

    async def _get(self, path: str, params: dict | None = None) -> dict:
        if not self.api_key:
            raise TMDBError(
                "TMDB_API_KEY is not set. Add it as a Repl Secret / .env value."
            )

        full_params = self._params(params)
        key = self._cache_key(path, full_params)

        now = time.monotonic()
        cached = self._cache.get(key)
        if cached and (now - cached[0]) < self.cache_ttl:
            return cached[1]

        client = await self._get_client()
        url = f"{self.base_url}{path}"

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await client.get(url, params=full_params)
            except httpx.HTTPError as e:
                # connection failures, timeouts, DNS errors, etc. - anything
                # httpx can raise before we even get a response. Retried a few
                # times before giving up.
                if attempt < self.max_retries:
                    await asyncio.sleep(self._retry_delay(attempt, None))
                    continue
                raise TMDBError(
                    f"TMDB {path} request failed after {self.max_retries} attempts: "
                    f"{type(e).__name__}: {e}"
                )

            if resp.status_code == 200:
                data = resp.json()
                self._cache[key] = (time.monotonic(), data)
                return data

            if resp.status_code == 404:
                raise TMDBNotFoundError(f"TMDB {path} not found (404): {resp.text}")

            if resp.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                await asyncio.sleep(self._retry_delay(attempt, resp))
                continue

            # 400 and 404 (and other non-retryable statuses) surface immediately.
            raise TMDBError(f"TMDB {path} failed: {resp.status_code} {resp.text}")

        raise TMDBError(f"TMDB {path} request failed after {self.max_retries} attempts")

    async def search_movies(self, query: str, page: int = 1, year: int | None = None) -> dict:
        params: dict = {"query": query, "page": page}
        if year:
            params["year"] = year
        return await self._get("/search/movie", params)

    async def get_movie(self, movie_id: int) -> dict:
        # append_to_response pulls credits, keywords, videos, and
        # watch/providers in a single call, saving three extra round trips
        # per movie. The response is additive: every key the old response had
        # is still present, plus `videos` and `watch/providers` blocks.
        return await self._get(
            f"/movie/{movie_id}",
            {"append_to_response": "credits,keywords,videos,watch/providers"},
        )

    async def get_reviews(self, movie_id: int, page: int = 1) -> dict:
        return await self._get(f"/movie/{movie_id}/reviews", {"page": page})

    async def get_popular(self, page: int = 1) -> dict:
        return await self._get("/movie/popular", {"page": page})

    async def discover_movies(self, params: dict | None = None) -> dict:
        """Flexible TMDB discovery used by natural-language Phase 4 search."""
        return await self._get("/discover/movie", params or {})

    async def get_recommendations(self, movie_id: int, page: int = 1) -> dict:
        return await self._get(f"/movie/{movie_id}/recommendations", {"page": page})


tmdb_client = TMDBClient()
