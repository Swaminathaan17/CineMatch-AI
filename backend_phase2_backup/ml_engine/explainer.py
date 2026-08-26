"""
Turns a raw similarity result (score + shared genres/cast/director) into
human-readable recommendation reasons. This is what makes a recommendation
feel explainable instead of just a bare percentage.

Kept as its own module since Week 2/3 will extend this to also factor in
sentiment and user-preference match - the reason list will just grow.
"""
from __future__ import annotations


def explain_content_match(match: dict) -> list[str]:
    reasons = []

    if match["shared_director"]:
        who = ", ".join(match["shared_director"])
        reasons.append(f"Same director: {who}")

    if match["shared_genres"]:
        genres = ", ".join(match["shared_genres"])
        reasons.append(f"Shares genres: {genres}")

    if match["shared_cast"]:
        cast = ", ".join(match["shared_cast"][:3])
        reasons.append(f"Shared cast: {cast}")

    pct = round(match["similarity_score"] * 100)
    reasons.append(f"{pct}% thematic similarity based on plot, genre, and crew")

    return reasons


def to_match_percentage(similarity_score: float) -> int:
    """Cosine similarity scores rarely hit 1.0 in practice for genuinely
    different movies, so a raw score reads as artificially low to a user
    (e.g. 0.31 looks unimpressive even for a strong match). Rescale onto a
    more legible 0-100 band without pretending precision we don't have."""
    return min(99, round(similarity_score * 180))
