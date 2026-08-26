"""Multi-signal recommendation scoring for Phase 3.

The ranker keeps every signal on a comparable 0-1 scale and explicitly handles
missing user/sentiment data instead of inventing values.
"""
from __future__ import annotations

from pathlib import Path
from datetime import date
import math

import yaml

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.yaml"


def load_weights() -> dict:
    with open(WEIGHTS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def recency_score(release_date: str | None, half_life_days: int = 3650) -> float:
    """Return a gentle freshness signal, 1.0 for today and ~0.5 at half-life."""
    if not release_date:
        return 0.0
    try:
        released = date.fromisoformat(str(release_date)[:10])
    except (ValueError, TypeError):
        return 0.0
    age = max(0, (date.today() - released).days)
    return round(math.exp(-math.log(2) * age / half_life_days), 4)


def compute_hybrid_score(
    content_similarity: float,
    sentiment_positive_pct: float | None,
    rating: float,
    popularity: float,
    popularity_max_in_batch: float,
    preference_match: float | None = None,
    recency: float | None = None,
    feedback_boost: float = 0.0,
    weights: dict | None = None,
) -> dict:
    """Blend content, audience, metadata, freshness and user taste.

    Missing signals have their configured weight redistributed over the signals
    that actually exist. feedback_boost is deliberately capped and acts as a
    small personalization nudge rather than overpowering content similarity.
    """
    w = weights or load_weights()
    signals = {
        "content_similarity": _normalize(content_similarity, 0, 1),
        "sentiment": None if sentiment_positive_pct is None else _normalize(sentiment_positive_pct, 0, 100),
        "rating": _normalize(rating, 0, 10),
        "popularity": _normalize(math.log1p(max(0.0, popularity)), 0, math.log1p(max(1.0, popularity_max_in_batch))),
        "preference_match": None if preference_match is None else _normalize(preference_match, 0, 1),
        "recency": 0.0 if recency is None else _normalize(recency, 0, 1),
    }

    available = {name: value for name, value in signals.items() if value is not None and name in w}
    total_weight = sum(w[name] for name in available) or 1.0
    final = sum((w[name] / total_weight) * value for name, value in available.items())
    feedback_boost = max(-0.08, min(0.08, feedback_boost))
    final = max(0.0, min(1.0, final + feedback_boost))

    components = {k: (None if v is None else round(v, 4)) for k, v in signals.items()}
    components["feedback_boost"] = round(feedback_boost, 4)
    return {"final_score": round(final, 4), "components": components}


def compute_confidence(
    content_similarity: float,
    has_sentiment: bool,
    review_count: int,
    popularity: float,
    popularity_max_in_batch: float,
) -> dict:
    content = max(0.0, min(1.0, content_similarity))
    review_signal = min(1.0, review_count / 50) if has_sentiment else 0.0
    pop_signal = _normalize(popularity, 0, popularity_max_in_batch or 1)
    ratio = 0.55 * content + 0.25 * review_signal + 0.20 * pop_signal
    label = "High" if ratio >= 0.75 else "Medium" if ratio >= 0.4 else "Low"
    return {"label": label, "score": round(ratio, 2)}
