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
    original = config.settings.tmdb_api_key
    yield
    config.settings.tmdb_api_key = original


def test_connection_failure_wraps_in_tmdberror_not_raw_exception():
    """Regression test: httpx exceptions (timeouts, connection failures) must
    be wrapped in TMDBError, not leaked raw - otherwise every caller's
    'except TMDBError' handling is bypassed and the request crashes with an
    unhandled 500 instead of degrading gracefully."""
    from app.services.tmdb_client import TMDBClient, TMDBError
    import asyncio

    async def run():
        client = TMDBClient()
        config.settings.tmdb_api_key = "fake_key_for_test"
        with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=httpx.ConnectTimeout("simulated"))):
            with pytest.raises(TMDBError):
                await client.search_movies("test")

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
