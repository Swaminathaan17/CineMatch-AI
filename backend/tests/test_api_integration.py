import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)  # triggers startup event (init_db) via context handling below


@pytest.fixture(autouse=True, scope="module")
def _ensure_startup():
    with TestClient(app):
        yield


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_list_movies_returns_array():
    res = client.get("/movies/")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_get_movie_not_found():
    res = client.get("/movies/999999")
    assert res.status_code == 404


@pytest.fixture(scope="module")
def sample_movie_id():
    """Grabs a real movie id from whatever dataset is actually loaded,
    rather than hardcoding one that only existed in the old sample data."""
    res = client.get("/movies/")
    movies = res.json()
    assert movies, "no movies in dataset - can't run movie-dependent tests"
    return movies[0]["id"]


def test_recommendations_for_known_movie(sample_movie_id):
    res = client.get(f"/recommendations/{sample_movie_id}?top_n=3")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body, list)
    assert len(body) <= 3
    if body:
        assert "match_percentage" in body[0]
        assert "reasons" in body[0]


def test_recommendations_for_unknown_movie_404():
    res = client.get("/recommendations/999999")
    assert res.status_code == 404


def test_personalized_cold_start_returns_trending_mode():
    res = client.get("/recommendations/personalized?session_id=pytest-cold-start-user")
    assert res.status_code == 200
    assert res.json()["mode"] == "trending"


def test_personalized_after_liking_a_movie(sample_movie_id):
    session_id = "pytest-warm-user"
    like_res = client.post(
        "/users/interactions",
        json={"session_id": session_id, "movie_id": sample_movie_id, "interaction_type": "liked"},
    )
    assert like_res.status_code == 200

    rec_res = client.get(f"/recommendations/personalized?session_id={session_id}")
    assert rec_res.status_code == 200
    assert rec_res.json()["mode"] == "personalized"


def test_discovery_query_returns_results():
    res = client.post("/discovery/query?top_n=3", json={"query": "a sci-fi movie about dreams"})
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] in ("semantic", "fallback")
    assert isinstance(body["results"], list)


def test_search_finds_exact_title(sample_movie_id):
    movie = client.get(f"/movies/{sample_movie_id}").json()
    title_fragment = movie["title"].split()[0]
    res = client.get(f"/movies/search?q={title_fragment}")
    assert res.status_code == 200
    body = res.json()
    assert body["found"] is True
    assert any(r["id"] == sample_movie_id for r in body["results"])


def test_search_nonexistent_movie_returns_honest_not_found():
    res = client.get("/movies/search?q=xyzqwerty_definitely_not_a_movie_title_zzz")
    assert res.status_code == 200
    body = res.json()
    assert body["found"] is False
    assert body["results"] == []
    assert "message" in body


def test_search_does_not_regress_to_422_routing_bug():
    """Regression test: /search and /search/tmdb must resolve before the
    /{movie_id} catch-all, or 'search' gets swallowed as a failed int
    conversion (422) instead of reaching these handlers."""
    res = client.get("/movies/search?q=test")
    assert res.status_code == 200
