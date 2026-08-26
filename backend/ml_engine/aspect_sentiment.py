"""
Aspect-based sentiment analysis.

Approach: split each review into sentences, anchor each sentence to one or
more aspects (story/acting/direction/visuals/music/pacing) using a curated
keyword dictionary, then run the anchored sentences through the same
sentiment model used for overall sentiment. This is a lightweight, explainable
approach - no separate model to train, and each aspect score is traceable
back to specific sentences if you want to show that in the UI.

Aspects with too few anchored sentences are marked "insufficient_data" rather
than guessing from a single stray sentence.
"""
from __future__ import annotations

import re
from collections import defaultdict

from ml_engine.sentiment_model import sentiment_model

ASPECT_KEYWORDS = {
    "story": [
        "story", "plot", "script", "screenplay", "writing", "narrative",
        "twist", "ending", "storyline", "premise",
    ],
    "acting": [
        "acting", "actor", "actress", "performance", "cast", "chemistry",
        "portrayal", "character",
    ],
    "direction": [
        "direction", "director", "directed", "editing", "edited", "shot",
        "filmmaking",
    ],
    "visuals": [
        "visuals", "cinematography", "effects", "cgi", "visual", "camera",
        "scenery", "production design", "costumes",
    ],
    "music": [
        "soundtrack", "score", "music", "song", "composer",
    ],
    "pacing": [
        "pacing", "pace", "slow", "dragged", "runtime", "length", "boring",
        "tedious",
    ],
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def anchor_sentences_to_aspects(sentences: list[str]) -> dict[str, list[str]]:
    """Returns {aspect: [sentence, sentence, ...]}. A sentence can anchor to
    more than one aspect if it mentions keywords from both."""
    anchored: dict[str, list[str]] = defaultdict(list)
    for sentence in sentences:
        lowered = sentence.lower()
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(kw in lowered for kw in keywords):
                anchored[aspect].append(sentence)
    return anchored


def _label_from_positive_ratio(positive_ratio: float) -> str:
    if positive_ratio >= 0.8:
        return "Very Positive"
    if positive_ratio >= 0.6:
        return "Positive"
    if positive_ratio >= 0.4:
        return "Mixed"
    if positive_ratio >= 0.2:
        return "Negative"
    return "Very Negative"


def analyze_aspects(reviews: list[str], min_sentences: int = 3) -> dict:
    """
    reviews: list of raw review text strings for one movie.
    Returns a dict keyed by aspect name, each value either:
        {"status": "scored", "label": "Positive", "positive_pct": 0.75, "sentence_count": 6}
        or
        {"status": "insufficient_data", "sentence_count": 1}
    """
    all_anchored: dict[str, list[str]] = defaultdict(list)
    for review in reviews:
        sentences = split_sentences(review)
        anchored = anchor_sentences_to_aspects(sentences)
        for aspect, sents in anchored.items():
            all_anchored[aspect].extend(sents)

    results = {}
    for aspect in ASPECT_KEYWORDS:
        sentences = all_anchored.get(aspect, [])
        if len(sentences) < min_sentences:
            results[aspect] = {
                "status": "insufficient_data",
                "sentence_count": len(sentences),
            }
            continue

        predictions = sentiment_model.predict_batch(sentences)
        positive_count = sum(1 for p in predictions if p.label == "positive")
        positive_ratio = positive_count / len(predictions)

        results[aspect] = {
            "status": "scored",
            "label": _label_from_positive_ratio(positive_ratio),
            "positive_pct": round(positive_ratio * 100, 1),
            "sentence_count": len(sentences),
        }
    return results


def analyze_overall(reviews: list[str]) -> dict:
    """Overall sentiment across all reviews (not aspect-specific)."""
    if not reviews:
        return {"status": "insufficient_data", "review_count": 0}

    predictions = sentiment_model.predict_batch(reviews)
    positive_count = sum(1 for p in predictions if p.label == "positive")
    total = len(predictions)
    positive_pct = round((positive_count / total) * 100, 1)
    negative_pct = round(100 - positive_pct, 1)

    return {
        "status": "scored",
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
        "review_count": total,
        "label": _label_from_positive_ratio(positive_count / total),
    }
