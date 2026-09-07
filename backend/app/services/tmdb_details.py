"""
Pure helpers that turn raw TMDB movie-detail sub-blocks (videos and
watch/providers, both fetched via append_to_response) into small, normalized
structures for the movie-detail API.

Kept separate from data_prep.py on purpose: data_prep reshapes TMDB JSON into
the flat row format ml_engine expects, while this module only enriches the
detail endpoint's response (trailer pick + streaming availability).
"""
from __future__ import annotations

YOUTUBE_URL = "https://www.youtube.com/watch?v={key}"

# Group keys TMDB uses inside a region's watch/provider payload.
PROVIDER_TYPES = ("flatrate", "rent", "buy", "ads")


def select_trailer(videos: list[dict] | None) -> dict | None:
    """Deterministically pick the best trailer from TMDB's `videos` block.

    Preference order:
      1. YouTube site
      2. type == "Trailer"
      3. officially listed trailer
      4. English-language video where available
    otherwise the first reasonable video.

    Returns None when there are no videos at all. Never makes extra requests.
    """
    if not videos:
        return None

    def priority(video: dict) -> tuple:
        return (
            0 if str(video.get("site", "")).lower() == "youtube" else 1,
            0 if str(video.get("type", "")).lower() == "trailer" else 1,
            0 if video.get("official") else 1,
            0 if str(video.get("iso_639_1", "")).lower() == "en" else 1,
        )

    best = min(videos, key=priority)
    key = best.get("key")
    is_youtube = str(best.get("site", "")).lower() == "youtube"
    return {
        "site": best.get("site"),
        "key": key,
        "name": best.get("name"),
        "type": best.get("type"),
        "official": bool(best.get("official")),
        "url": YOUTUBE_URL.format(key=key) if is_youtube and key else None,
    }


def parse_watch_providers(
    watch_data: dict | None, region: str | None = "US"
) -> dict | None:
    """Normalize TMDB's `watch/providers` block for a single region.

    `watch_data` is the raw block appended to the movie-detail call:
    {"id": ..., "results": {"US": {"link": ..., "flatrate": [...], ...}}}

    Returns a dict with the watch link plus provider lists grouped by
    availability type (flatrate/rent/buy/ads), or None when TMDB has no
    provider data at all (same for an unsupported region) so callers never
    claim a movie is streamable without TMDB actually saying so.
    """
    if not watch_data:
        return None

    region_key = (region or "US").upper()
    region_data = (watch_data.get("results") or {}).get(region_key)
    if not region_data:
        return None

    groups: dict[str, list[dict]] = {}
    for provider_type in PROVIDER_TYPES:
        providers = region_data.get(provider_type) or []
        groups[provider_type] = [
            {
                "provider_name": provider.get("provider_name"),
                "logo_path": provider.get("logo_path"),
                "display_priority": provider.get("display_priority"),
            }
            for provider in providers
        ]

    return {
        "region": region_key,
        "link": region_data.get("link"),
        "available": any(bool(groups[t]) for t in PROVIDER_TYPES),
        **groups,
    }