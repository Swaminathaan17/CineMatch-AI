"""Human-readable reasons for recommendation scores."""
from __future__ import annotations


def explain_content_match(match: dict) -> list[str]:
    reasons = []
    if match.get("shared_director"):
        reasons.append(f"Same director: {', '.join(match['shared_director'])}")
    if match.get("shared_genres"):
        reasons.append(f"Shares genres: {', '.join(match['shared_genres'])}")
    if match.get("shared_cast"):
        reasons.append(f"Shared cast: {', '.join(match['shared_cast'][:3])}")
    pct = round(match.get("similarity_score", 0) * 100)
    reasons.append(f"{pct}% thematic similarity based on plot, genre, and crew")
    return reasons


def explain_hybrid_match(match: dict) -> list[str]:
    reasons = explain_content_match(match)
    components = match.get("score_breakdown", {})
    pref = components.get("preference_match")
    rating = components.get("rating")
    recency = components.get("recency")
    sentiment = components.get("sentiment")
    if pref is not None and pref >= 0.65:
        reasons.append("Strong match with your taste profile")
    if rating is not None and rating >= 0.75:
        reasons.append("Highly rated by audiences")
    if sentiment is not None and sentiment >= 0.70:
        reasons.append("Strong positive audience sentiment")
    if recency is not None and recency >= 0.80:
        reasons.append("Relatively recent release")
    return reasons[:6]


def to_match_percentage(similarity_score: float) -> int:
    return min(99, max(0, round(similarity_score * 180)))
