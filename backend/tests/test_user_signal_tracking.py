"""
Phase 2 Step 2.1 - real user-signal tracking.

Covers the persistence/API behaviour of the five tracked signals:
likes, views, ratings, searches, and watchlist add/remove.

Likes are idempotent (one active like per user+movie, timestamp refreshed).
Views are preserved as separate behavioral events.
Ratings are 1-5 with one row per user+movie (re-rate overwrites).
Searches preserve history but dedupe rapid repeats of the same query.
Watchlist adds/removes are additionally logged to watchlist_events.
"""
import sys
import time
import uuid
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import user_service

client = TestClient(app)


@pytest.fixture(autouse=True, scope="module")
def _ensure_startup():
    with TestClient(app):
        yield


@pytest.fixture(scope="module")
def sample_movie_id():
    res = client.get("/movies/")
    return res.json()[0]["id"]


def _liked_rows(session_id):
    body = client.get(f"/users/interactions?session_id={session_id}").json()
    return [r for r in body["results"] if r["interaction_type"] == "liked"]


def test_like_is_idempotent_and_refreshable(sample_movie_id):
    session_id = f"pytest-like-{uuid.uuid4().hex}"
    client.post(
        "/users/interactions",
        json={"session_id": session_id, "movie_id": sample_movie_id, "interaction_type": "liked"},
    )
    first_ts = _liked_rows(session_id)[0]["created_at"]
    time.sleep(0.02)
    client.post(
        "/users/interactions",
        json={"session_id": session_id, "movie_id": sample_movie_id, "interaction_type": "liked"},
    )
    rows = _liked_rows(session_id)
    assert len(rows) == 1
    assert rows[0]["created_at"] > first_ts


def test_views_are_preserved_as_separate_events(sample_movie_id):
    session_id = f"pytest-view-{uuid.uuid4().hex}"
    for _ in range(3):
        client.post(
            "/users/interactions",
            json={"session_id": session_id, "movie_id": sample_movie_id, "interaction_type": "viewed"},
        )
    body = client.get(f"/users/interactions?session_id={session_id}").json()
    viewed = [r for r in body["results"] if r["interaction_type"] == "viewed"]
    assert len(viewed) == 3


def test_rating_upsert_overwrites_value(sample_movie_id):
    session_id = f"pytest-rating-{uuid.uuid4().hex}"
    for rating in (3, 5):
        res = client.post(
            "/users/ratings",
            json={"session_id": session_id, "movie_id": sample_movie_id, "rating": rating},
        )
        assert res.status_code == 200
    ratings = client.get(f"/users/ratings?session_id={session_id}").json()["ratings"]
    assert ratings[str(sample_movie_id)] == 5.0


@pytest.mark.parametrize("bad_rating", [0, -1, 5.5, 6])
def test_rating_rejects_values_outside_1_5(bad_rating, sample_movie_id):
    session_id = f"pytest-rating-bad-{uuid.uuid4().hex}"
    res = client.post(
        "/users/ratings",
        json={"session_id": session_id, "movie_id": sample_movie_id, "rating": bad_rating},
    )
    assert res.status_code == 400


def test_search_preserves_history_but_dedupes_rapid_repeats():
    session_id = f"pytest-search-{uuid.uuid4().hex}"
    query = f"interstellar dreams {uuid.uuid4().hex}"
    for _ in range(3):
        res = client.post(
            "/users/searches",
            json={"session_id": session_id, "query": query, "result_source": "local", "result_count": 5},
        )
        assert res.status_code == 200

    body = client.get(f"/users/searches?session_id={session_id}").json()
    matches = [r for r in body["results"] if r["query"] == query]
    assert len(matches) == 1
    assert matches[0]["result_count"] == 5

    other_query = f"another query {uuid.uuid4().hex}"
    client.post("/users/searches", json={"session_id": session_id, "query": other_query})
    body2 = client.get(f"/users/searches?session_id={session_id}").json()
    assert len(body2["results"]) == 2


def test_search_expired_window_creates_new_row(monkeypatch):
    session_id = f"pytest-search-exp-{uuid.uuid4().hex}"
    query = f"expired query {uuid.uuid4().hex}"
    client.post("/users/searches", json={"session_id": session_id, "query": query})
    time.sleep(0.01)
    monkeypatch.setattr(user_service, "SEARCH_DEDUPE_WINDOW", timedelta(0))
    client.post("/users/searches", json={"session_id": session_id, "query": query})
    body = client.get(f"/users/searches?session_id={session_id}").json()
    matches = [r for r in body["results"] if r["query"] == query]
    assert len(matches) == 2


def test_watchlist_add_remove_logs_history_and_state_stays_clean(sample_movie_id):
    session_id = f"pytest-wl-event-{uuid.uuid4().hex}"
    client.post(
        "/users/watchlist",
        json={"session_id": session_id, "movie_id": sample_movie_id, "movie_title": "Signal Movie"},
    )
    # Idempotent re-add must not log a second event.
    client.post(
        "/users/watchlist",
        json={"session_id": session_id, "movie_id": sample_movie_id, "movie_title": "Signal Movie"},
    )
    client.delete(f"/users/watchlist/{sample_movie_id}?session_id={session_id}")

    events = client.get(f"/users/watchlist/events?session_id={session_id}").json()["results"]
    actions = [(e["action"], e["movie_title"]) for e in events]
    assert actions == [("removed", "Signal Movie"), ("added", "Signal Movie")]

    wl = client.get(f"/users/watchlist?session_id={session_id}").json()["results"]
    assert all(r["id"] != sample_movie_id for r in wl)