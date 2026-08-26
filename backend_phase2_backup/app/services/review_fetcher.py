"""
Fetches reviews for sentiment analysis. Primary source is TMDB's own
/movie/{id}/reviews endpoint (official API, no scraping fragility).

TMDB review volume per movie varies a lot - popular films can have dozens,
obscure ones often have zero. When a movie has fewer reviews than
settings.min_reviews_for_sentiment, callers should surface "insufficient_data"
rather than compute a misleading score off 1-2 reviews.
"""
from __future__ import annotations

from app.services.tmdb_client import tmdb_client, TMDBError


async def fetch_reviews(movie_id: int, max_pages: int = 3) -> list[str]:
    """Returns raw review text strings for a movie, up to max_pages of results."""
    texts: list[str] = []
    try:
        page = 1
        while page <= max_pages:
            data = await tmdb_client.get_reviews(movie_id, page=page)
            results = data.get("results", [])
            texts.extend([r["content"] for r in results if r.get("content")])
            if page >= data.get("total_pages", 1):
                break
            page += 1
    except TMDBError:
        # network/API issue - return whatever we got (possibly empty),
        # let the caller's insufficient_data path handle it rather than
        # crashing the whole recommendation request over a reviews fetch
        pass
    return texts
