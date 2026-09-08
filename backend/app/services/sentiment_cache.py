"""In-process TTL cache for computed sentiment / review-analysis results.

Process-local by design: the app runs under a single uvicorn worker, mirroring
the existing tmdb_client._cache pattern. The cache resets on restart, so no
stored result can ever go stale forever, and nothing is written to the
SQLite database (movie_rec.db stays untouched).

Keys are tuples built from movie_id plus every input that changes the result:
the analysis version (algorithm / model changes), the review source, and the
minimum-review threshold. Bumping SENTIMENT_ANALYSIS_VERSION lazily invalidates
every old entry without any schema or migration work.

Stored values are the full public result dictionaries, optionally carrying an
internal "_review_count" metadata key used by the hybrid recommendation flow
(so its confidence estimate stays exact without a redundant TMDB call).
Only SUCCESSFUL analyses are cached: failures and non-dict payloads are never
stored, and expired or structurally corrupt entries are treated as misses and
discarded rather than raised.

get/set are synchronous and complete within a single event-loop turn, so no
locking is required in the single-process app.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

SENTIMENT_ANALYSIS_VERSION = "v1"
DEFAULT_SENTIMENT_REVIEW_SOURCE = "tmdb"
DEFAULT_TTL_SECONDS = 24 * 60 * 60  # 24h


class SentimentCache:
    def __init__(self, ttl: float = DEFAULT_TTL_SECONDS, now: Optional[Callable[[], float]] = None):
        self.ttl = ttl
        self._now = now or time.monotonic
        self._entries: dict[tuple, tuple[float, dict]] = {}

    def build_key(
        self,
        movie_id: int,
        version: str = SENTIMENT_ANALYSIS_VERSION,
        source: str = DEFAULT_SENTIMENT_REVIEW_SOURCE,
        min_reviews: int = 0,
    ) -> tuple:
        if isinstance(movie_id, str):
            movie_id = int(movie_id)
        return (
            int(movie_id),
            str(version),
            str(source),
            int(min_reviews),
        )

    def get(self, key: tuple) -> Optional[dict]:
        try:
            stored_at, value = self._entries[key]
        except (KeyError, TypeError, ValueError):
            return None
        if not isinstance(value, dict) or not value:
            self._entries.pop(key, None)
            return None
        try:
            expired = self._now() - stored_at >= self.ttl
        except TypeError:
            expired = True
        if expired:
            self._entries.pop(key, None)
            return None
        return value

    def set(self, key: tuple, value: dict) -> None:
        if not isinstance(value, dict) or not value:
            return
        self._entries[key] = (self._now(), value)

    def clear(self) -> None:
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def stats(self) -> dict:
        return {
            "entries": len(self._entries),
            "analysis_version": SENTIMENT_ANALYSIS_VERSION,
            "ttl_seconds": self.ttl,
        }


def as_public_payload(payload: dict) -> dict:
    """Strip internal underscore-prefixed metadata keys so API responses stay
    byte-identical to the pre-cache shapes."""
    return {k: v for k, v in payload.items() if not str(k).startswith("_")}


sentiment_cache = SentimentCache()