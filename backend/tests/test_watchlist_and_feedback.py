import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True, scope="module")
def _ensure_startup():
    with TestClient(app):
        yield


@pytest.fixture(scope="module")
def sample_movie_id():
    res = client.get("/movies/")
    return res.json()[0]["id"]


def test_watchlist_add_list_remove(sample_movie_id):
    session_id = "pytest-watchlist-user"

    add_res = client.post(
        "/users/watchlist",
        json={"session_id": session_id, "movie_id": sample_movie_id, "movie_title": "Test Movie"},
    )
    assert add_res.status_code == 200

    list_res = client.get(f"/users/watchlist?session_id={session_id}")
    assert list_res.status_code == 200
    ids = [r["id"] for r in list_res.json()["results"]]
    assert sample_movie_id in ids

    del_res = client.delete(f"/users/watchlist/{sample_movie_id}?session_id={session_id}")
    assert del_res.status_code == 200

    list_res2 = client.get(f"/users/watchlist?session_id={session_id}")
    ids2 = [r["id"] for r in list_res2.json()["results"]]
    assert sample_movie_id not in ids2


def test_watchlist_add_is_idempotent(sample_movie_id):
    session_id = "pytest-watchlist-idempotent"
    for _ in range(3):
        res = client.post(
            "/users/watchlist",
            json={"session_id": session_id, "movie_id": sample_movie_id, "movie_title": "Test"},
        )
        assert res.status_code == 200

    list_res = client.get(f"/users/watchlist?session_id={session_id}")
    matching = [r for r in list_res.json()["results"] if r["id"] == sample_movie_id]
    assert len(matching) == 1  # not duplicated across repeated adds


def test_feedback_downvote_excludes_from_future_hybrid_recommendations(sample_movie_id):
    session_id = "pytest-feedback-user"

    initial = client.get(f"/recommendations/{sample_movie_id}/hybrid?top_n=5&candidate_pool=10").json()
    if not initial:
        pytest.skip("no recommendations available for this movie to test against")
    target = initial[0]["id"]

    feedback_res = client.post(
        "/recommendations/feedback",
        json={
            "session_id": session_id,
            "source_movie_id": sample_movie_id,
            "recommended_movie_id": target,
            "feedback": "down",
        },
    )
    assert feedback_res.status_code == 200

    after = client.get(
        f"/recommendations/{sample_movie_id}/hybrid?top_n=5&candidate_pool=10&session_id={session_id}"
    ).json()
    after_ids = [r["id"] for r in after]
    assert target not in after_ids


def test_feedback_rejects_invalid_value(sample_movie_id):
    res = client.post(
        "/recommendations/feedback",
        json={
            "session_id": "pytest-bad-feedback",
            "source_movie_id": sample_movie_id,
            "recommended_movie_id": sample_movie_id,
            "feedback": "sideways",
        },
    )
    assert res.status_code == 400


def test_hybrid_recommendations_include_confidence_field(sample_movie_id):
    res = client.get(f"/recommendations/{sample_movie_id}/hybrid?top_n=3&candidate_pool=5")
    assert res.status_code == 200
    body = res.json()
    if body:
        assert "confidence" in body[0]
        assert body[0]["confidence"]["label"] in ("High", "Medium", "Low")
        assert 0 <= body[0]["confidence"]["score"] <= 1


def test_feedback_is_upserted_not_duplicated(sample_movie_id):
    """Repeated feedback on the same (user, source, recommended) triple must stay
    a single DB row whose value reflects the latest submission."""
    from uuid import uuid4

    from app.db.models import RecommendationFeedback, User
    from app.db.session import SessionLocal

    session_id = f"pytest-fb-upsert-{uuid4().hex}"
    target = sample_movie_id + 1

    for _ in range(3):
        res = client.post(
            "/recommendations/feedback",
            json={
                "session_id": session_id,
                "source_movie_id": sample_movie_id,
                "recommended_movie_id": target,
                "feedback": "up",
            },
        )
        assert res.status_code == 200

    res = client.post(
        "/recommendations/feedback",
        json={
            "session_id": session_id,
            "source_movie_id": sample_movie_id,
            "recommended_movie_id": target,
            "feedback": "down",
        },
    )
    assert res.status_code == 200

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.session_id == session_id).first()
        assert user is not None
        rows = (
            db.query(RecommendationFeedback)
            .filter(
                RecommendationFeedback.user_id == user.id,
                RecommendationFeedback.source_movie_id == sample_movie_id,
                RecommendationFeedback.recommended_movie_id == target,
            )
            .all()
        )
        assert len(rows) == 1  # upserted, never duplicated
        assert rows[0].feedback == "down"  # latest submission wins
    finally:
        user = db.query(User).filter(User.session_id == session_id).first()
        if user:
            db.query(RecommendationFeedback).filter(
                RecommendationFeedback.user_id == user.id
            ).delete(synchronize_session=False)
            db.query(User).filter(User.id == user.id).delete(synchronize_session=False)
        db.commit()
        db.close()
