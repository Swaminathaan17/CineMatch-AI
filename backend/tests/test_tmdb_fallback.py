import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import config


@pytest.fixture(autouse=True)
def _reset_tmdb_key():
    from app.services.tmdb_client import tmdb_client as shared_client

    original = config.settings.tmdb_api_key
    shared_client.clear_cache()
    yield
    config.settings.tmdb_api_key = original
    shared_client.clear_cache()


def test_connection_failure_wraps_in_tmdberror_not_raw_exception():
    """Regression test: httpx exceptions (timeouts, connection failures) must
    be wrapped in TMDBError, not leaked raw - otherwise every caller's
    'except TMDBError' handling is bypassed and the request crashes with an
    unhandled 500 instead of degrading gracefully."""
    from app.services.tmdb_client import TMDBClient, TMDBError
    import asyncio

    async def run():
        client = TMDBClient(max_retries=2, retry_base_delay=0.01, retry_max_delay=0.01)
        config.settings.tmdb_api_key = "fake_key_for_test"
        with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=httpx.ConnectTimeout("simulated"))) as mock_get:
            with pytest.raises(TMDBError):
                await client.search_movies("test")
            assert mock_get.call_count == 2  # initial attempt + 1 retry, then give up

    asyncio.run(run())


def test_not_found_error_is_distinguished_from_service_error():
    """404s and service errors are distinct: TMDBNotFoundError is a TMDBError
    (so existing 'except TMDBError' callers still catch it) but is not the
    same class, letting callers special-case 'resource missing'."""
    from app.services.tmdb_client import TMDBError, TMDBNotFoundError

    assert issubclass(TMDBNotFoundError, TMDBError)
    assert TMDBNotFoundError.__name__ != "TMDBError"


def test_retries_transient_status_then_succeeds():
    """A transient 5xx should be retried (exponential backoff) and, if a later
    attempt succeeds, the call returns the payload rather than erroring."""
    from app.services.tmdb_client import TMDBClient
    import asyncio

    async def run():
        client = TMDBClient(max_retries=3, retry_base_delay=0.01, retry_max_delay=0.01)
        config.settings.tmdb_api_key = "fake_key_for_test"
        req = httpx.Request("GET", "http://tmdb")
        flaky = AsyncMock(
            side_effect=[
                httpx.Response(503, request=req, json={}),
                httpx.Response(502, request=req, json={}),
                httpx.Response(200, request=req, json={"results": [{"id": 1}]}),
            ]
        )
        with patch.object(httpx.AsyncClient, "get", flaky):
            data = await client.search_movies("retry_me")
        assert data == {"results": [{"id": 1}]}
        assert flaky.call_count == 3

    asyncio.run(run())


def test_429_respects_retry_after_then_succeeds():
    """429 should honor Retry-After and eventually succeed."""
    from app.services.tmdb_client import TMDBClient
    import asyncio

    async def run():
        client = TMDBClient(max_retries=3, retry_base_delay=0.01, retry_max_delay=0.01)
        config.settings.tmdb_api_key = "fake_key_for_test"
        req = httpx.Request("GET", "http://tmdb")
        rate_limited = AsyncMock(
            side_effect=[
                httpx.Response(429, request=req, headers={"Retry-After": "0"}, json={}),
                httpx.Response(200, request=req, json={"results": [{"id": 7}]}),
            ]
        )
        with patch.object(httpx.AsyncClient, "get", rate_limited):
            data = await client.search_movies("rate_limited_query")
        assert data == {"results": [{"id": 7}]}
        assert rate_limited.call_count == 2

    asyncio.run(run())


def test_no_retry_on_400():
    """A hard 400 is a caller/protocol error - must not be retried."""
    from app.services.tmdb_client import TMDBClient, TMDBError
    import asyncio

    async def run():
        client = TMDBClient(max_retries=3, retry_base_delay=0.01, retry_max_delay=0.01)
        config.settings.tmdb_api_key = "fake_key_for_test"
        req = httpx.Request("GET", "http://tmdb")
        bad = AsyncMock(return_value=httpx.Response(400, request=req, text="bad request"))
        with patch.object(httpx.AsyncClient, "get", bad):
            with pytest.raises(TMDBError):
                await client.search_movies("bad_query")
        assert bad.call_count == 1

    asyncio.run(run())


def test_no_retry_on_404_and_raises_not_found_error():
    """A 404 is not retried and is raised as TMDBNotFoundError (a TMDBError
    subclass) so callers can distinguish 'not found' from service failures."""
    from app.services.tmdb_client import TMDBClient, TMDBNotFoundError, TMDBError
    import asyncio

    async def run():
        client = TMDBClient(max_retries=3, retry_base_delay=0.01, retry_max_delay=0.01)
        config.settings.tmdb_api_key = "fake_key_for_test"
        req = httpx.Request("GET", "http://tmdb")
        missing = AsyncMock(return_value=httpx.Response(404, request=req, text="not found"))
        with patch.object(httpx.AsyncClient, "get", missing):
            with pytest.raises(TMDBNotFoundError):
                await client.get_movie(123456789)
        assert missing.call_count == 1

    asyncio.run(run())


def test_successful_get_is_cached_within_ttl():
    """Successful GET responses are cached: a second identical call within the
    TTL must not hit the network again."""
    from app.services.tmdb_client import TMDBClient
    import asyncio

    async def run():
        client = TMDBClient(max_retries=1, cache_ttl=60)
        config.settings.tmdb_api_key = "fake_key_for_test"
        req = httpx.Request("GET", "http://tmdb")
        ok = AsyncMock(return_value=httpx.Response(200, request=req, json={"results": [{"id": 3}]}))
        with patch.object(httpx.AsyncClient, "get", ok):
            first = await client.search_movies("cached_query")
            second = await client.search_movies("cached_query")
        assert first == second == {"results": [{"id": 3}]}
        assert ok.call_count == 1

    asyncio.run(run())


def test_cache_respects_ttl():
    """Once the TTL expires the cache must re-query the network."""
    from app.services.tmdb_client import TMDBClient
    import asyncio

    async def run():
        client = TMDBClient(max_retries=1, cache_ttl=0.05)
        config.settings.tmdb_api_key = "fake_key_for_test"
        req = httpx.Request("GET", "http://tmdb")
        ok = AsyncMock(return_value=httpx.Response(200, request=req, json={"results": [{"id": 4}]}))
        with patch.object(httpx.AsyncClient, "get", ok):
            await client.search_movies("ttl_query")
            await asyncio.sleep(0.06)
            await client.search_movies("ttl_query")
        assert ok.call_count == 2

    asyncio.run(run())


def test_clear_cache_forces_new_request():
    """clear_cache() must drop cached responses so the next call re-fetches."""
    from app.services.tmdb_client import TMDBClient
    import asyncio

    async def run():
        client = TMDBClient(max_retries=1, cache_ttl=60)
        config.settings.tmdb_api_key = "fake_key_for_test"
        req = httpx.Request("GET", "http://tmdb")
        ok = AsyncMock(return_value=httpx.Response(200, request=req, json={"results": [{"id": 5}]}))
        with patch.object(httpx.AsyncClient, "get", ok):
            await client.search_movies("clear_cache_query")
            client.clear_cache()
            await client.search_movies("clear_cache_query")
        assert ok.call_count == 2

    asyncio.run(run())


def test_search_falls_back_to_tmdb_when_not_found_locally_and_key_present():
    config.settings.tmdb_api_key = "fake_key_for_test"

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "results": [{"id": 496243, "title": "Parasite"}]
    }

    with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=fake_response)):
        with TestClient(app) as client:
            res = client.get("/movies/search?q=xyz_definitely_not_local_zzz")
            assert res.status_code == 200
            body = res.json()
            assert body["found"] is True
            assert body["source"] == "tmdb"
            assert body["results"][0]["title"] == "Parasite"
            assert "note" in body  # honest disclosure that recs won't work for this title


def test_search_degrades_gracefully_when_tmdb_unreachable():
    """Regression test: when a key is configured but TMDB can't be reached,
    the endpoint must still return 200 with an honest message - not a raw
    500 from an unwrapped exception (this was a real bug caught during
    manual testing before this test existed)."""
    config.settings.tmdb_api_key = "fake_key_for_test"

    with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=httpx.ConnectTimeout("simulated"))):
        with TestClient(app) as client:
            res = client.get("/movies/search?q=xyz_definitely_not_local_zzz")
            assert res.status_code == 200
            body = res.json()
            assert body["found"] is False
            assert body["results"] == []


def test_search_without_key_does_not_attempt_tmdb_call():
    config.settings.tmdb_api_key = ""
    with TestClient(app) as client:
        res = client.get("/movies/search?q=xyz_definitely_not_local_zzz")
        assert res.status_code == 200
        body = res.json()
        assert body["found"] is False
        assert "no TMDB key" in body["message"]


def test_tmdb_movie_can_be_used_as_external_recommendation_query():
    """Phase 1: a TMDB-only movie should produce recommendations from the
    local catalogue without being added to the local dataset."""
    config.settings.tmdb_api_key = "fake_key_for_test"

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "id": 999001,
        "title": "External Sci-Fi Movie",
        "overview": "A crew travels through space.",
        "genres": [{"name": "Science Fiction"}],
        "credits": {
            "cast": [{"name": "Test Actor"}],
            "crew": [{"name": "Test Director", "job": "Director"}],
        },
        "keywords": {"keywords": [{"name": "space"}]},
        "vote_average": 8.0,
        "popularity": 50.0,
    }

    with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=fake_response)):
        with TestClient(app) as client:
            res = client.get("/recommendations/tmdb/999001?top_n=3")
            assert res.status_code == 200
            body = res.json()
            assert isinstance(body, list)
            assert len(body) <= 3
            if body:
                assert "match_percentage" in body[0]
                assert "reasons" in body[0]
                assert body[0]["id"] != 999001


def test_tmdb_movie_can_be_imported_into_persistent_library_and_removed():
    """Phase 2: importing a TMDB movie persists it, puts it into the TF-IDF
    catalogue, and makes it behave like a normal recommendation source."""
    config.settings.tmdb_api_key = "fake_key_for_phase2"
    movie_id = 999002

    from app.db.session import SessionLocal
    from app.db.models import LibraryMovie
    from app.services.recommendation_service import recommendation_service

    # Keep the test deterministic if it is re-run in the same SQLite database.
    db = SessionLocal()
    try:
        existing = db.get(LibraryMovie, movie_id)
        if existing:
            db.delete(existing)
            db.commit()
    finally:
        db.close()
    recommendation_service.refresh()

    fake_detail = {
        "id": movie_id,
        "title": "Phase Two Space Movie",
        "overview": "A crew travels through deep space.",
        "genres": [{"name": "Science Fiction"}],
        "credits": {
            "cast": [{"name": "Test Actor"}],
            "crew": [{"name": "Test Director", "job": "Director"}],
        },
        "keywords": {"keywords": [{"name": "space"}, {"name": "survival"}]},
        "poster_path": "/phase2.jpg",
        "backdrop_path": "/phase2-backdrop.jpg",
        "release_date": "2026-01-01",
        "vote_average": 8.2,
        "popularity": 42.0,
    }

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = fake_detail

    with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=fake_response)):
        with TestClient(app) as client:
            add = client.post(f"/movies/{movie_id}/library")
            assert add.status_code == 200
            body = add.json()
            assert body["status"] == "added"
            assert body["movie"]["source"] == "tmdb"

            # It survives a full in-memory index rebuild.
            persisted = client.get(f"/movies/{movie_id}")
            assert persisted.status_code == 200
            assert persisted.json()["title"] == "Phase Two Space Movie"
            assert persisted.json()["source"] == "tmdb"

            # It is now a first-class source movie for the recommender, rather
            # than the Phase 1 temporary-query path.
            recs = client.get(f"/recommendations/{movie_id}?top_n=3")
            assert recs.status_code == 200
            assert all(r["id"] != movie_id for r in recs.json())

            # Importing again is idempotent and refreshes metadata.
            add_again = client.post(f"/movies/{movie_id}/library")
            assert add_again.status_code == 200
            assert add_again.json()["status"] == "updated"

            remove = client.delete(f"/movies/{movie_id}/library")
            assert remove.status_code == 200
            assert remove.json()["status"] == "removed"

            missing = client.get(f"/movies/{movie_id}")
            assert missing.status_code == 404
