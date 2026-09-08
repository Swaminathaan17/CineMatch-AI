"""
Phase 2 Step 2.2 - personalization from real user signals.

Uses a small synthetic movie set so these tests never trigger the full
(6,856-movie) ContentSimilarityEngine warm-up. Covers the recommendation
engine's behavior-signal scoring: likes, ratings, views, watchlist, recent
searches, negative feedback, recency decay, double-count caps, and batched
signal retrieval.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
from sqlalchemy import event
from uuid import uuid4

from ml_engine.content_similarity import ContentSimilarityEngine
from ml_engine.personalization import PersonalizationEngine, signal_decay


def make_df():
    return pd.DataFrame([
        {"id": 1, "title": "Deep Space", "overview": "astronaut survives alone in space", "genres": "Science Fiction, Drama", "cast": "A", "director": "Nolan", "keywords": "space, survival"},
        {"id": 2, "title": "Mars Survival", "overview": "astronaut fights to survive alone on mars", "genres": "Science Fiction, Drama", "cast": "B", "director": "Scott", "keywords": "mars, survival"},
        {"id": 3, "title": "Space Comedy", "overview": "crew has a funny adventure in space", "genres": "Science Fiction, Comedy", "cast": "C", "director": "Lee", "keywords": "space, comedy"},
        {"id": 4, "title": "Romantic Wedding", "overview": "friends prepare for a romantic wedding", "genres": "Romance", "cast": "D", "director": "Smith", "keywords": "wedding, love"},
        {"id": 5, "title": "Alien Contact", "overview": "crew encounters an alien on a deep space vessel", "genres": "Science Fiction, Thriller", "cast": "E", "director": "Cameron", "keywords": "alien, space"},
        {"id": 6, "title": "Wedding Planner", "overview": "best friends plan a busy wedding party", "genres": "Romance, Comedy", "cast": "F", "director": "Jones", "keywords": "wedding, party"},
    ])


@pytest.fixture(scope="module")
def personal():
    engine = ContentSimilarityEngine(make_df())
    return PersonalizationEngine(engine)


def _now(days_ago=0):
    return datetime.now(timezone.utc) - timedelta(days=days_ago)


def _bundle(**overrides):
    base = {
        "liked": [], "ratings": [], "watchlist": [], "views": [],
        "feedback": [], "favorite_genres": [], "searches": [],
        "liked_ids": [], "viewed_ids": [], "watchlist_ids": [], "downvoted": set(),
    }
    base.update(overrides)
    return base


def test_signal_decay_prefers_recent_activity():
    fresh = signal_decay(_now(2))
    older = signal_decay(_now(120))
    ancient = signal_decay(_now(600))
    assert older < fresh
    assert ancient <= older
    assert signal_decay(_now(600)) >= 0.1  # floor keeps a small influence


def test_liked_movie_increases_relevant_preference(personal):
    liked = _bundle(liked=[(1, _now())], liked_ids=[1])
    empty = _bundle()
    assert personal.behavior_scores(empty) == {}
    scores = personal.behavior_scores(liked)
    assert scores[2] > scores[4]  # similar Sci-Fi boosted, unrelated movie not


def test_high_rating_increases_preference_rating_scaled(personal):
    strong = _bundle(ratings=[(1, 5.0, _now())])
    neutral = _bundle(ratings=[(1, 3.0, _now())])
    weak = _bundle(ratings=[(1, 1.0, _now())])
    s_strong = personal.behavior_scores(strong)[2]
    s_neutral = personal.behavior_scores(neutral)[2]
    s_weak = personal.behavior_scores(weak)[2]
    assert s_strong > s_neutral
    assert s_neutral > s_weak


def test_low_rating_decreases_preference(personal):
    positive = _bundle(ratings=[(1, 5.0, _now())])
    negative = _bundle(ratings=[(1, 1.0, _now())])
    assert personal.behavior_scores(positive)[2] > personal.behavior_scores(negative)[2]
    weights = personal.aggregate_signal_weights(negative)
    assert weights[1] < 0


def test_recent_signals_beat_old_signals(personal):
    recent = _bundle(liked=[(1, _now(3))], liked_ids=[1])
    old = _bundle(liked=[(1, _now(400))], liked_ids=[1])
    assert personal.behavior_scores(recent)[2] > personal.behavior_scores(old)[2]


def test_views_weaker_than_likes_but_repeats_matter(personal):
    liked = _bundle(liked=[(1, _now())], liked_ids=[1])
    one_view = _bundle(views=[(1, _now())])
    three_views = _bundle(views=[(1, _now())] * 3)
    s_like = personal.behavior_scores(liked)[2]
    s_one = personal.behavior_scores(one_view)[2]
    s_three = personal.behavior_scores(three_views)[2]
    assert s_one < s_three < s_like


def test_watchlist_contributes_positive_preference(personal):
    watch = _bundle(watchlist=[(1, _now())], watchlist_ids=[1])
    one_view = _bundle(views=[(1, _now())])
    assert personal.behavior_scores(watch)[2] > personal.behavior_scores(one_view)[2]


def test_negative_feedback_remains_effective(personal):
    positive_only = _bundle(liked=[(1, _now())], liked_ids=[1])
    with_negative = _bundle(
        liked=[(1, _now())], liked_ids=[1],
        feedback=[(3, "down", _now())], downvoted={3},
    )
    assert personal.behavior_scores(with_negative)[2] <= personal.behavior_scores(positive_only)[2]
    boosts = personal.movie_behavior_boosts(with_negative)
    assert boosts[3] < 0
    assert personal.aggregate_signal_weights(with_negative)[3] < 0


def test_no_signals_cold_start(personal):
    empty = _bundle()
    assert personal.behavior_scores(empty) == {}
    assert personal.movie_behavior_boosts(empty) == {}
    assert personal.has_enough_behavioral_signal(empty) is False


def test_repeated_signals_capped_not_double_counted(personal):
    stacked = _bundle(
        liked=[(1, _now())], liked_ids=[1],
        ratings=[(1, 5.0, _now())],
        watchlist=[(1, _now())], watchlist_ids=[1],
        views=[(1, _now())] * 3,
    )
    weight = personal.aggregate_signal_weights(stacked)[1]
    assert abs(weight) <= 1.5 + 1e-9       # capped, not stacked to ~2.95
    assert weight > 1.0                    # but clearly stronger than like alone


def test_rating_high_value_movies_shifts_genre_preference(personal):
    sci_fi_fan = _bundle(ratings=[(1, 5.0, _now()), (3, 5.0, _now())])
    scores = personal.behavior_scores(sci_fi_fan)
    assert scores[5] > scores[4]  # Sci-Fi movie preferred over Romance


def test_recent_search_hints_prefer_genres(personal):
    base = _bundle(views=[(1, _now())])
    with_search = _bundle(views=[(1, _now())], searches=[("a sci-fi thriller with a twist", _now())])
    gap_no = personal.behavior_scores(base)[5] - personal.behavior_scores(base)[4]
    gap_yes = personal.behavior_scores(with_search)[5] - personal.behavior_scores(with_search)[4]
    assert gap_yes > gap_no  # search hint tilts the profile toward Sci-Fi/Thriller

    search_only = _bundle(searches=[("a sci-fi thriller with a twist", _now())])
    assert personal.behavior_scores(search_only) == {}  # no movie-scale signal yet


def test_signal_bundle_retrieval_is_batched():
    """Signals must come from a small fixed number of queries, never once per
    candidate movie."""
    from app.db.session import SessionLocal, engine
    from app.services import user_service
    from app.db import models

    session_id = f"pytest-sig-bundle-batch-{uuid4()}"
    db = SessionLocal()
    try:
        user_service.get_or_create_user(db, session_id)
        user_service.rate_movie(db, session_id, 1, 5)
        user_service.record_search_query(db, session_id, "sci-fi")
        user_service.add_to_watchlist(db, session_id, 2, "Mars")
        user_service.record_interaction(db, session_id, 3, "liked")
        user_service.record_interaction(db, session_id, 4, "viewed")

        count = {"n": 0}

        def count_selects(conn, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                count["n"] += 1

        event.listen(engine, "before_cursor_execute", count_selects)
        try:
            bundle = user_service.get_signal_bundle(db, session_id)
        finally:
            event.remove(engine, "before_cursor_execute", count_selects)
    finally:
        user = db.query(models.User).filter(models.User.session_id == session_id).first()
        if user:
            db.query(models.UserSearchQuery).filter(
                models.UserSearchQuery.user_id == user.id
            ).delete(synchronize_session=False)
            db.query(models.UserInteraction).filter(
                models.UserInteraction.user_id == user.id
            ).delete(synchronize_session=False)
            db.query(models.MovieRating).filter(
                models.MovieRating.user_id == user.id
            ).delete(synchronize_session=False)
            db.query(models.WatchlistItem).filter(
                models.WatchlistItem.user_id == user.id
            ).delete(synchronize_session=False)
            db.query(models.WatchlistEvent).filter(
                models.WatchlistEvent.user_id == user.id
            ).delete(synchronize_session=False)
            db.query(models.User).filter(models.User.id == user.id).delete(
                synchronize_session=False
            )
        for pfx in ("pytest-sig-bundle-batch", "pytest-sig-bundle-batch-"):
            leftover = db.query(models.User).filter(
                models.User.session_id.like(f"{pfx}%")
            ).all()
            for u in leftover:
                db.query(models.UserSearchQuery).filter(
                    models.UserSearchQuery.user_id == u.id
                ).delete(synchronize_session=False)
                db.query(models.UserInteraction).filter(
                    models.UserInteraction.user_id == u.id
                ).delete(synchronize_session=False)
                db.query(models.MovieRating).filter(
                    models.MovieRating.user_id == u.id
                ).delete(synchronize_session=False)
                db.query(models.WatchlistItem).filter(
                    models.WatchlistItem.user_id == u.id
                ).delete(synchronize_session=False)
                db.query(models.WatchlistEvent).filter(
                    models.WatchlistEvent.user_id == u.id
                ).delete(synchronize_session=False)
                db.query(models.User).filter(models.User.id == u.id).delete(
                    synchronize_session=False
                )
        db.commit()
        db.close()

    assert count["n"] <= 12
    assert len(bundle["ratings"]) >= 1
    assert bundle["liked_ids"] == [3]
    assert 4 in set(bundle["viewed_ids"])
    assert 2 in set(bundle["watchlist_ids"])