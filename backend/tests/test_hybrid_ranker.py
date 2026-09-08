import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_engine.hybrid_ranker import compute_hybrid_score


def test_hybrid_score_within_bounds_with_full_signals():
    result = compute_hybrid_score(
        content_similarity=0.5, sentiment_positive_pct=70, rating=7.5,
        popularity=80, popularity_max_in_batch=100,
    )
    assert 0 <= result["final_score"] <= 1


def test_missing_sentiment_does_not_default_to_neutral():
    """When sentiment is None, its component in the output must also be None -
    not silently treated as 0.5, which would fabricate a signal we don't have."""
    result = compute_hybrid_score(
        content_similarity=0.5, sentiment_positive_pct=None, rating=7.5,
        popularity=80, popularity_max_in_batch=100,
    )
    assert result["components"]["sentiment"] is None


def test_missing_sentiment_redistributes_weight_higher_than_zero_fill():
    """Redistributing weight away from a missing signal should score a movie
    higher than treating the missing signal as the worst case (0)."""
    with_sentiment = compute_hybrid_score(
        content_similarity=0.8, sentiment_positive_pct=0, rating=9, popularity=100,
        popularity_max_in_batch=100,
    )
    without_sentiment = compute_hybrid_score(
        content_similarity=0.8, sentiment_positive_pct=None, rating=9, popularity=100,
        popularity_max_in_batch=100,
    )
    assert without_sentiment["final_score"] > with_sentiment["final_score"]


def test_higher_content_similarity_increases_score_all_else_equal():
    low = compute_hybrid_score(
        content_similarity=0.1, sentiment_positive_pct=50, rating=5,
        popularity=50, popularity_max_in_batch=100,
    )
    high = compute_hybrid_score(
        content_similarity=0.9, sentiment_positive_pct=50, rating=5,
        popularity=50, popularity_max_in_batch=100,
    )
    assert high["final_score"] > low["final_score"]


def test_zero_popularity_max_does_not_divide_by_zero():
    result = compute_hybrid_score(
        content_similarity=0.5, sentiment_positive_pct=50, rating=5,
        popularity=0, popularity_max_in_batch=0,
    )
    assert result["final_score"] >= 0


def test_preference_signal_can_change_rank_score():
    low = compute_hybrid_score(
        content_similarity=0.5, sentiment_positive_pct=70, rating=7,
        popularity=50, popularity_max_in_batch=100, preference_match=0.1,
        recency=0.5,
    )
    high = compute_hybrid_score(
        content_similarity=0.5, sentiment_positive_pct=70, rating=7,
        popularity=50, popularity_max_in_batch=100, preference_match=0.9,
        recency=0.5,
    )
    assert high["final_score"] > low["final_score"]


def test_missing_preference_is_not_fabricated():
    result = compute_hybrid_score(
        content_similarity=0.7, sentiment_positive_pct=70, rating=7,
        popularity=50, popularity_max_in_batch=100, preference_match=None,
        recency=0.5,
    )
    assert result["components"]["preference_match"] is None


def test_recency_is_bounded_and_recent_is_stronger():
    from ml_engine.hybrid_ranker import recency_score
    assert 0 <= recency_score("2026-08-20") <= 1
    assert recency_score("2026-08-20") > recency_score("2015-01-01")


def test_popularity_log_scaling_does_not_let_extreme_outlier_dominate():
    from ml_engine.hybrid_ranker import compute_hybrid_score
    moderate = compute_hybrid_score(0.8, None, 8, 100, 10000)
    extreme = compute_hybrid_score(0.5, None, 6, 10000, 10000)
    assert moderate["final_score"] > extreme["final_score"]
