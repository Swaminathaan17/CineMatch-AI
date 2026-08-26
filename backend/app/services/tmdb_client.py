"""
Thin wrapper around the TMDB API. Every call goes through here so caching
and error handling live in one place instead of being scattered across routers.
"""
import httpx
from app.config import settings


class TMDBError(Exception):
    pass


class TMDBClient:
    def __init__(self):
        self.base_url = settings.tmdb_base_url

    @property
    def api_key(self) -> str:
        # read dynamically rather than capturing once at construction time -
        # this module is imported (and the singleton below created) at
        # startup, so a static self.api_key would silently ignore any key
        # set after that point, which also made this class hard to test
        return settings.tmdb_api_key

    def _params(self, extra: dict | None = None) -> dict:
        params = {"api_key": self.api_key}
        if extra:
            params.update(extra)
        return params

    async def _get(self, path: str, params: dict | None = None) -> dict:
        if not self.api_key:
            raise TMDBError(
                "TMDB_API_KEY is not set. Add it as a Repl Secret / .env value."
            )
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.base_url}{path}", params=self._params(params)
                )
        except httpx.HTTPError as e:
            # covers connection failures, timeouts, DNS errors, etc. - anything
            # httpx can raise before we even get a response. Without this,
            # these exceptions leak past every caller's "except TMDBError"
            # handling and crash the endpoint with a raw 500 instead of
            # falling back gracefully.
            raise TMDBError(f"TMDB {path} request failed: {type(e).__name__}: {e}")

        if resp.status_code != 200:
            raise TMDBError(f"TMDB {path} failed: {resp.status_code} {resp.text}")
        return resp.json()

    async def search_movies(self, query: str, page: int = 1) -> dict:
        return await self._get("/search/movie", {"query": query, "page": page})

    async def get_movie(self, movie_id: int) -> dict:
        # append_to_response pulls credits + keywords in a single call,
        # saving an extra round trip per movie
        return await self._get(
            f"/movie/{movie_id}", {"append_to_response": "credits,keywords"}
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
