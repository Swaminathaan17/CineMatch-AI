import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.sentiment_cache import SentimentCache, as_public_payload, SENTIMENT_ANALYSIS_VERSION

FAKE_OVERALL = {"status": "scored", "positive_pct": 80.0, "label": "Positive"}
FAKE_ASPECTS = {"acting": {"status": "scored", "positive_pct": 90.0, "label": "Very Positive"}}


@pytest.fixture(autouse=True)
def _clean_sentiment_cache():
    from app.services.sentiment_cache import sentiment_cache as shared

    shared.clear()
    yield
    shared.clear()


# ---------------------------------------------------------------------------
# Unit tests: SentimentCache keying, TTL, corruption safety
# ---------------------------------------------------------------------------


def test_build_key_is_deterministic_and_coerces_types():
    cache = SentimentCache()
    assert cache.build_key(42, "v1", "tmdb", 3) == cache.build_key("42", "v1", "tmdb", "3")
    assert cache.build_key(42) == (42, SENTIMENT_ANALYSIS_VERSION, "tmdb", 0)


def test_miss_returns_none():
    cache = SentimentCache()
    assert cache.get(cache.build_key(1)) is None


def test_set_then_get_roundtrip():
    cache = SentimentCache()
    key = cache.build_key(7)
    cache.set(key, {"overall": FAKE_OVERALL})
    assert cache.get(key) == {"overall": FAKE_OVERALL}


def test_keys_isolate_by_movie_id():
    cache = SentimentCache()
    cache.set(cache.build_key(1), {"movie": 1})
    cache.set(cache.build_key(2), {"movie": 2})
    assert cache.get(cache.build_key(1)) == {"movie": 1}
    assert cache.get(cache.build_key(2)) == {"movie": 2}
    assert cache.size == 2


def test_keys_isolate_by_analysis_version():
    cache = SentimentCache()
    cache.set(cache.build_key(1, version="v1"), {"ver": "v1"})
    assert cache.get(cache.build_key(1, version="v2")) is None
    assert cache.get(cache.build_key(1, version="v1")) == {"ver": "v1"}


def test_keys_isolate_by_review_source():
    cache = SentimentCache()
    cache.set(cache.build_key(1, source="tmdb"), {"src": "tmdb"})
    assert cache.get(cache.build_key(1, source="other")) is None


def test_keys_isolate_by_min_reviews_threshold():
    cache = SentimentCache()
    cache.set(cache.build_key(1, min_reviews=3), {"min": 3})
    cache.set(cache.build_key(1, min_reviews=5), {"min": 5})
    assert cache.get(cache.build_key(1, min_reviews=3)) == {"min": 3}
    assert cache.get(cache.build_key(1, min_reviews=5)) == {"min": 5}


def test_expired_entry_treated_as_miss_and_discarded():
    clock = {"t": 0.0}

    def now():
        return clock["t"]

    cache = SentimentCache(ttl=10.0, now=now)
    key = cache.build_key(1)
    cache.set(key, {"overall": FAKE_OVERALL})
    assert cache.get(key) == {"overall": FAKE_OVERALL}
    clock["t"] = 10.0
    assert cache.get(key) is None
    assert cache.size == 0


def test_corrupt_value_treated_as_miss_without_raising():
    cache = SentimentCache(ttl=60.0)
    key = cache.build_key(1)
    cache.set(key, {"overall": FAKE_OVERALL})
    cache._entries[key] = ("junk", [1, 2, 3])
    assert cache.get(key) is None
    assert cache.size == 0
    cache.set(key, {"overall": FAKE_OVERALL})
    assert cache.get(key) == {"overall": FAKE_OVERALL}


def test_corrupt_timestamp_treated_as_miss_without_raising():
    cache = SentimentCache(ttl=60.0)
    key = cache.build_key(2)
    cache._entries[key] = ("not-a-number", {"overall": FAKE_OVERALL})
    assert cache.get(key) is None
    assert cache.size == 0


def test_set_refuses_non_dict_values():
    cache = SentimentCache()
    key = cache.build_key(3)
    cache.set(key, None)
    cache.set(key, [])
    cache.set(key, {})
    cache.set(key, "text")
    cache.set(key, 42)
    assert cache.size == 0
    assert cache.get(key) is None


def test_set_overwrites_same_key():
    cache = SentimentCache()
    key = cache.build_key(4)
    cache.set(key, {"overall": FAKE_OVERALL})
    cache.set(key, {"status": "insufficient_data", "review_count": 0})
    assert cache.get(key) == {"status": "insufficient_data", "review_count": 0}
    assert cache.size == 1


def test_repeated_misses_coalesce_into_single_entry_after_set():
    cache = SentimentCache()
    key = cache.build_key(777)
    assert cache.get(key) is None
    assert cache.get(key) is None
    cache.set(key, {"movie_id": 777})
    assert cache.size == 1
    assert cache.get(key) == {"movie_id": 777}


def test_clear_empties_cache():
    cache = SentimentCache()
    cache.set(cache.build_key(1), {"a": 1})
    cache.set(cache.build_key(2), {"b": 2})
    cache.clear()
    assert cache.size == 0
    assert cache.get(cache.build_key(1)) is None


def test_stats_report_version_and_ttl():
    cache = SentimentCache(ttl=123)
    stats = cache.stats
    assert stats["analysis_version"] == SENTIMENT_ANALYSIS_VERSION
    assert stats["ttl_seconds"] == 123
    assert cache.stats["entries"] == 0


def test_as_public_payload_strips_internal_keys():
    payload = {"movie_id": 9, "overall": FAKE_OVERALL, "_review_count": 5}
    assert as_public_payload(payload) == {"movie_id": 9, "overall": FAKE_OVERALL}


# ---------------------------------------------------------------------------
# Helpers for endpoint-level integration
# ---------------------------------------------------------------------------


def _make_counters():
    calls = {"fetch": 0, "overall": 0, "aspects": 0}

    async def fake_fetch(movie_id):
        calls["fetch"] += 1
        return ["Great film.", "Loved it!", "Solid.", "Awesome.", "Good."]

    def fake_overall(reviews):
        calls["overall"] += 1
        return FAKE_OVERALL

    def fake_aspects(reviews, min_sentences=3):
        calls["aspects"] += 1
        return FAKE_ASPECTS

    return calls, fake_fetch, fake_overall, fake_aspects


def _patched_sentiment_routes(monkeypatch, calls, fetch, overall, aspects):
    import app.routers.sentiment as sentiment_router

    monkeypatch.setattr(sentiment_router, "fetch_reviews", fetch)
    monkeypatch.setattr(sentiment_router, "analyze_overall", overall)
    monkeypatch.setattr(sentiment_router, "analyze_aspects", aspects)


def _patched_hybrid_routes(monkeypatch, calls, fetch, overall, aspects):
    import app.routers.recommendations as rec_router
    from app.services.recommendation_service import recommendation_service

    monkeypatch.setattr(rec_router, "fetch_reviews", fetch)
    monkeypatch.setattr(rec_router, "analyze_overall", overall)
    monkeypatch.setattr(rec_router, "analyze_aspects", aspects)

    def fake_get_similar(movie_id, top_n=10):
        return [
            {"id": 55601, "title": "Alpha", "similarity_score": 0.9},
            {"id": 55602, "title": "Beta", "similarity_score": 0.8},
        ]

    def fake_row(cid):
        return {"vote_average": 8.0, "popularity": 50.0, "release_date": "2020-06-01"}

    def fake_diversify(items, top_n=10):
        return items[:top_n]

    monkeypatch.setattr(recommendation_service, "get_similar", fake_get_similar)
    monkeypatch.setattr(recommendation_service, "get_movie_row", fake_row)
    monkeypatch.setattr(recommendation_service, "diversify", fake_diversify)


# ---------------------------------------------------------------------------
# Endpoint integration: GET /sentiment/{movie_id}
# ---------------------------------------------------------------------------


def test_sentiment_endpoint_caches_success_and_skips_recompute(monkeypatch):
    calls, fetch, overall, aspects = _make_counters()
    _patched_sentiment_routes(monkeypatch, calls, fetch, overall, aspects)

    with TestClient(app) as client:
        first = client.get("/sentiment/55501")
        second = client.get("/sentiment/55501")

    assert first.status_code == 200
    assert first.json() == {"movie_id": 55501, "overall": FAKE_OVERALL, "aspects": FAKE_ASPECTS}
    assert second.json() == first.json()
    assert calls == {"fetch": 1, "overall": 1, "aspects": 1}


def test_insufficient_data_result_is_cached(monkeypatch):
    calls = {"fetch": 0}

    async def empty_fetch(movie_id):
        calls["fetch"] += 1
        return []

    _patched_sentiment_routes(monkeypatch, calls, empty_fetch, lambda r: {}, lambda r, min_sentences=3: {})

    with TestClient(app) as client:
        first = client.get("/sentiment/55502")
        second = client.get("/sentiment/55502")

    body = first.json()
    assert set(body) == {"status", "review_count", "message"}
    assert body["status"] == "insufficient_data"
    assert body["message"] == (
        "Only 0 review(s) found - need at least 3 for a reliable score."
    )
    assert calls["fetch"] == 1
    assert second.json() == body


def test_endpoint_movies_do_not_share_cache_entries(monkeypatch):
    calls, fetch, overall, aspects = _make_counters()
    _patched_sentiment_routes(monkeypatch, calls, fetch, overall, aspects)

    with TestClient(app) as client:
        client.get("/sentiment/55503")
        client.get("/sentiment/55504")

    assert calls == {"fetch": 2, "overall": 2, "aspects": 2}


def test_expired_entry_is_refetched_by_endpoint(monkeypatch):
    calls, fetch, overall, aspects = _make_counters()
    _patched_sentiment_routes(monkeypatch, calls, fetch, overall, aspects)

    import app.routers.sentiment as sentiment_router

    fresh = SentimentCache(ttl=0.05)
    monkeypatch.setattr(sentiment_router, "sentiment_cache", fresh)

    with TestClient(app) as client:
        first = client.get("/sentiment/55505")
        assert first.status_code == 200
        time.sleep(0.06)
        second = client.get("/sentiment/55505")

    assert second.status_code == 200
    assert calls == {"fetch": 2, "overall": 2, "aspects": 2}


def test_failed_analysis_is_not_cached_and_retried(monkeypatch):
    from ml_engine.sentiment_model import SentimentModelNotTrainedError
    from app.services.sentiment_cache import sentiment_cache as shared

    calls = {"fetch": 0}

    async def fake_fetch(movie_id):
        calls["fetch"] += 1
        return ["a", "b", "c", "d", "e"]

    def failing_overall(reviews):
        raise SentimentModelNotTrainedError("model not trained")

    import app.routers.sentiment as sentiment_router

    monkeypatch.setattr(sentiment_router, "fetch_reviews", fake_fetch)
    monkeypatch.setattr(sentiment_router, "analyze_overall", failing_overall)
    monkeypatch.setattr(sentiment_router, "analyze_aspects", lambda r, min_sentences=3: FAKE_ASPECTS)

    with TestClient(app) as client:
        first = client.get("/sentiment/55506")
        assert first.status_code == 503
        second = client.get("/sentiment/55506")
        assert second.status_code == 503

    assert calls["fetch"] == 2
    assert shared.size == 0


def test_sentiment_calls_query_no_database(monkeypatch):
    calls, fetch, overall, aspects = _make_counters()
    _patched_sentiment_routes(monkeypatch, calls, fetch, overall, aspects)

    from sqlalchemy.orm import Session as SA_Session

    executed = []
    original_execute = SA_Session.execute

    def spy_execute(self, *a, **k):
        executed.append(1)
        return original_execute(self, *a, **k)

    monkeypatch.setattr(SA_Session, "execute", spy_execute)

    with TestClient(app) as client:
        client.get("/sentiment/55507")
        client.get("/sentiment/55508")

    assert executed == []


# ---------------------------------------------------------------------------
# Endpoint integration: hybrid recommendation flow shares the cache
# ---------------------------------------------------------------------------


def test_hybrid_reuses_cached_sentiment_without_refetch(monkeypatch):
    sentiment_calls = {"fetch": 0, "overall": 0, "aspects": 0}
    hybrid_calls = {"fetch": 0, "overall": 0, "aspects": 0}

    async def sent_fetch(movie_id):
        sentiment_calls["fetch"] += 1
        return ["a", "b", "c", "d", "e"]

    def sent_overall(reviews):
        sentiment_calls["overall"] += 1
        return FAKE_OVERALL

    def sent_aspects(reviews, min_sentences=3):
        sentiment_calls["aspects"] += 1
        return FAKE_ASPECTS

    _patched_sentiment_routes(monkeypatch, sentiment_calls, sent_fetch, sent_overall, sent_aspects)

    async def hy_fetch(movie_id):
        hybrid_calls["fetch"] += 1
        return ["x", "y", "z"]

    def hy_overall(reviews):
        hybrid_calls["overall"] += 1
        return FAKE_OVERALL

    def hy_aspects(reviews, min_sentences=3):
        hybrid_calls["aspects"] += 1
        return FAKE_ASPECTS

    _patched_hybrid_routes(monkeypatch, hybrid_calls, hy_fetch, hy_overall, hy_aspects)

    with TestClient(app) as client:
        warm = client.get("/sentiment/55601")
        assert warm.status_code == 200
        res = client.get("/recommendations/55603/hybrid?top_n=2")
        assert res.status_code == 200

    body = res.json()
    assert len(body) == 2
    alphas = [r for r in body if r["id"] == 55601][0]
    assert alphas["match_percentage"] > 0
    assert any("positive audience sentiment" in reason for reason in alphas["reasons"])
    assert "confidence" in alphas
    # Only the non-cached candidate needed a fetch + analysis.
    assert hybrid_calls == {"fetch": 1, "overall": 1, "aspects": 1}
    assert sentiment_calls == {"fetch": 1, "overall": 1, "aspects": 1}


def test_hybrid_miss_stores_full_payload_serving_sentiment_endpoint(monkeypatch):
    sentiment_calls = {"fetch": 0}
    hybrid_calls = {"fetch": 0, "overall": 0, "aspects": 0}

    async def hy_fetch(movie_id):
        hybrid_calls["fetch"] += 1
        return ["a", "b", "c"]

    def hy_overall(reviews):
        hybrid_calls["overall"] += 1
        return FAKE_OVERALL

    def hy_aspects(reviews, min_sentences=3):
        hybrid_calls["aspects"] += 1
        return FAKE_ASPECTS

    _patched_hybrid_routes(monkeypatch, hybrid_calls, hy_fetch, hy_overall, hy_aspects)

    async def sent_fetch(movie_id):
        sentiment_calls["fetch"] += 1
        raise AssertionError("must be a cache hit - fetch should not happen")

    _patched_sentiment_routes(monkeypatch, sentiment_calls, sent_fetch, lambda r: {}, lambda r, min_sentences=3: {})

    with TestClient(app) as client:
        hy = client.get("/recommendations/55609/hybrid?top_n=2")
        assert hy.status_code == 200
        after = client.get("/sentiment/55602")
        assert after.status_code == 200

    assert after.json() == {"movie_id": 55602, "overall": FAKE_OVERALL, "aspects": FAKE_ASPECTS}
    assert sentiment_calls["fetch"] == 0
    # Both candidates were cache misses, so both were fetched + analyzed once.
    assert hybrid_calls == {"fetch": 2, "overall": 2, "aspects": 2}


def test_hybrid_cached_insufficient_data_does_not_refetch(monkeypatch):
    sentiment_calls = {"fetch": 0}
    hybrid_calls = {"fetch": 0}

    async def sent_fetch(movie_id):
        sentiment_calls["fetch"] += 1
        return []

    _patched_sentiment_routes(monkeypatch, sentiment_calls, sent_fetch, lambda r: {}, lambda r, min_sentences=3: {})

    async def hy_fetch(movie_id):
        hybrid_calls["fetch"] += 1
        return ["a", "b", "c"]

    _patched_hybrid_routes(monkeypatch, hybrid_calls, hy_fetch, lambda r: FAKE_OVERALL, lambda r, min_sentences=3: FAKE_ASPECTS)

    with TestClient(app) as client:
        ins = client.get("/sentiment/55601")
        assert ins.status_code == 200
        assert ins.json()["status"] == "insufficient_data"
        res = client.get("/recommendations/55603/hybrid?top_n=2")
        assert res.status_code == 200

    alphas = [r for r in res.json() if r["id"] == 55601][0]
    assert any("insufficient review data" in reason for reason in alphas["reasons"])
    # Cached insufficient_data for 55601 -> only 55602 was fetched.
    assert hybrid_calls["fetch"] == 1
    assert sentiment_calls["fetch"] == 1


def test_hybrid_cache_path_adds_no_database_queries(monkeypatch):
    sentiment_calls = {"fetch": 0, "overall": 0, "aspects": 0}
    hybrid_calls = {"fetch": 0, "overall": 0, "aspects": 0}

    async def sent_fetch(movie_id):
        sentiment_calls["fetch"] += 1
        return ["a", "b", "c", "d", "e"]

    _patched_sentiment_routes(monkeypatch, sentiment_calls, sent_fetch, lambda r: FAKE_OVERALL, lambda r, min_sentences=3: FAKE_ASPECTS)

    async def hy_fetch(movie_id):
        hybrid_calls["fetch"] += 1
        return ["a", "b", "c"]

    _patched_hybrid_routes(monkeypatch, hybrid_calls, hy_fetch, lambda r: FAKE_OVERALL, lambda r, min_sentences=3: FAKE_ASPECTS)

    from sqlalchemy.orm import Session as SA_Session

    executed = []
    original_execute = SA_Session.execute

    def spy_execute(self, *a, **k):
        executed.append(1)
        return original_execute(self, *a, **k)

    monkeypatch.setattr(SA_Session, "execute", spy_execute)

    with TestClient(app) as client:
        warm = client.get("/sentiment/55601")
        assert warm.status_code == 200
        res = client.get("/recommendations/55603/hybrid?top_n=2")
        assert res.status_code == 200

    assert executed == []