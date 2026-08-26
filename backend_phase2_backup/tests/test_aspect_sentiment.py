import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_engine import aspect_sentiment
from ml_engine.sentiment_model import SentimentResult


def test_split_sentences_basic():
    text = "The acting was great. The plot dragged though! Was it worth it?"
    sentences = aspect_sentiment.split_sentences(text)
    assert len(sentences) == 3


def test_anchor_sentence_to_multiple_aspects():
    sentences = ["The story and pacing were both a mess."]
    anchored = aspect_sentiment.anchor_sentences_to_aspects(sentences)
    assert "story" in anchored
    assert "pacing" in anchored


def test_aspect_below_threshold_is_insufficient_data():
    reviews = ["Great acting all around."]  # only 1 acting-related sentence
    result = aspect_sentiment.analyze_aspects(reviews, min_sentences=3)
    assert result["acting"]["status"] == "insufficient_data"


def test_aspect_scoring_uses_stubbed_model(monkeypatch):
    """Verifies the aggregation math (positive_pct, label) is correct,
    independent of how good the underlying trained model is."""

    def fake_predict_batch(sentences):
        # every sentence about acting is "positive" in this stub
        return [SentimentResult(label="positive", confidence=0.9) for _ in sentences]

    monkeypatch.setattr(aspect_sentiment.sentiment_model, "predict_batch", fake_predict_batch)

    reviews = [
        "The acting was great, truly great performances.",
        "Every actor delivered a strong performance here.",
        "The cast's acting was the highlight of the film.",
    ]
    result = aspect_sentiment.analyze_aspects(reviews, min_sentences=3)
    assert result["acting"]["status"] == "scored"
    assert result["acting"]["positive_pct"] == 100.0
    assert result["acting"]["label"] == "Very Positive"


def test_overall_sentiment_empty_reviews_returns_insufficient_data():
    result = aspect_sentiment.analyze_overall([])
    assert result["status"] == "insufficient_data"
