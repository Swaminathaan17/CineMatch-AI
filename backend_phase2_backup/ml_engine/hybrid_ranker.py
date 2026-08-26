"""
Combines multiple signals into one final recommendation score:

    final_score = w1*content_similarity + w2*sentiment + w3*rating + w4*popularity

(user_preference_match gets added as a fifth term in Week 3, once
personalization exists - the weights.yaml file already reserves room for it
so adding it later doesn't require restructuring this module.)

Weights live in weights.yaml, not in this file, so tuning them is a config
change you can point to as an experiment, not a code change buried in logic.
"""
from __future__ import annotations

from pathlib import Path

import yaml

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.yaml"


def load_weights() -> dict:
    with open(WEIGHTS_PATH) as f:
        return yaml.safe_load(f)


def _normalize(value: float, min_val: float, max_val: float) -> float:
    """Squash any raw signal onto 0-1 so different scales (0-10 ratings vs
    0-1 similarity vs uncapped popularity) can be combined fairly."""
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def compute_hybrid_score(
    content_similarity: float,
    sentiment_positive_pct: float | None,
    rating: float,
    popularity: float,
    popularity_max_in_batch: float,
    weights: dict | None = None,
) -> dict:
    """
    All inputs are raw values from their respective sources:
      content_similarity: 0-1 cosine similarity
      sentiment_positive_pct: 0-100, or None if insufficient review data
      rating: TMDB vote_average, 0-10
      popularity: TMDB popularity score (uncapped, dataset-relative)
      popularity_max_in_batch: the max popularity value across the current
        candidate set, used to normalize popularity relative to this batch
        rather than against an arbitrary global constant

    Returns the final score plus each normalized component, so the
    explainability layer can show which signal contributed what.
    """
    w = weights or load_weights()

    norm_content = _normalize(content_similarity, 0, 1)
    norm_rating = _normalize(rating, 0, 10)
    norm_popularity = _normalize(popularity, 0, popularity_max_in_batch or 1)

    if sentiment_positive_pct is None:
        # no review data - redistribute sentiment's weight proportionally
        # across the other signals rather than silently treating missing
        # data as "neutral" (0.5), which would be a fabricated assumption
        remaining = w["content_similarity"] + w["rating"] + w["popularity"]
        final = (
            (w["content_similarity"] / remaining) * norm_content
            + (w["rating"] / remaining) * norm_rating
            + (w["popularity"] / remaining) * norm_popularity
        )
        sentiment_component = None
    else:
        norm_sentiment = _normalize(sentiment_positive_pct, 0, 100)
        final = (
            w["content_similarity"] * norm_content
            + w["sentiment"] * norm_sentiment
            + w["rating"] * norm_rating
            + w["popularity"] * norm_popularity
        )
        sentiment_component = round(norm_sentiment, 4)

    return {
        "final_score": round(final, 4),
        "components": {
            "content_similarity": round(norm_content, 4),
            "sentiment": sentiment_component,
            "rating": round(norm_rating, 4),
            "popularity": round(norm_popularity, 4),
        },
    }


def compute_confidence(
    content_similarity: float,
    has_sentiment: bool,
    review_count: int,
    popularity: float,
    popularity_max_in_batch: float,
) -> dict:
    """
    How much should the user trust this specific recommendation? Not the
    same thing as match_percentage (how similar/good the match is) - a movie
    can be a strong thematic match but low-confidence if we have almost no
    review data or it's an obscure title, versus a well-known movie with
    plenty of reviews backing the sentiment score.

    Signals combined:
      - content_similarity: a very low raw similarity (below ~0.15) means
        even the best signal we have is weak
      - has_sentiment / review_count: more reviews = more reliable sentiment
        contribution, zero reviews = one fewer signal backing the score at all
      - popularity relative to the batch: a very obscure title in the
        candidate pool has less data behind its rating/popularity numbers
        generally (fewer people have rated/reviewed it)
    """
    score = 0.0
    max_score = 3.0

    if content_similarity >= 0.35:
        score += 1.0
    elif content_similarity >= 0.15:
        score += 0.5

    if has_sentiment:
        if review_count >= 10:
            score += 1.0
        elif review_count >= 3:
            score += 0.5
    # zero contribution if no sentiment data at all - honestly reflects that
    # one of our signals is missing for this recommendation

    normalized_popularity = _normalize(popularity, 0, popularity_max_in_batch or 1)
    if normalized_popularity >= 0.3:
        score += 1.0
    elif normalized_popularity >= 0.1:
        score += 0.5

    ratio = score / max_score
    if ratio >= 0.75:
        label = "High"
    elif ratio >= 0.4:
        label = "Medium"
    else:
        label = "Low"

    return {"label": label, "score": round(ratio, 2)}
