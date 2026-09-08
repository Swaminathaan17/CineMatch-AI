"""
Phase 2 Step 2.3 - "Why this movie?" explanations.

Uses the same small synthetic movie set as the Phase 2.2 tests so nothing here
warms up the full (6,856-movie) engine. Verifies every explanation maps to a
real signal that moved the Phase 2.2 score, respects the 1-primary + 1-secondary
cap, is deterministic, never exposes raw search text, never fabricates positive
claims from negative feedback, keeps existing response fields intact, and does
not add database queries beyond the already-batched bundle load.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uuid import uuid4

import pytest
import pandas as pd
from sqlalchemy import event

from ml_engine.content_similarity import ContentSimilarityEngine
from ml_engine.personalization import PersonalizationEngine
from ml_engine.recommendation_explainer import build_recommendation_explanations


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


def _explain(personal, bundle, results=None, mode="personalized"):
    """Run the whole explanation layer with the ranking context computed once."""
    weights = personal.aggregate_signal_weights(bundle)
    prefs = personal.behavior_scores(bundle)
    boosts = personal.movie_behavior_boosts(bundle)
    results = results or [{"id": 2, "title": "Mars Survival"}]
    return build_recommendation_explanations(
        results, personal, bundle, weights=weights,
        preference_scores=prefs, behavior_boosts=boosts, mode=mode,
    )


def _texts(expls):
    return [e["primary"] + (f" • {e['secondary']}" if e["secondary"] else "") for e in expls]


def test_liked_similar_movie_references_liked_movie(personal):
    bundle = _bundle(liked=[(1, _now())], liked_ids=[1])
    expls = _explain(personal, bundle)
    assert expls[0]["primary"] == "Because you liked Deep Space"
    # a genre preference learned from that like may follow as the single secondary
    assert expls[0]["secondary"] in (
        None,
        "Because you often watch Science Fiction",
    )


def test_highly_rated_similar_movie_references_actual_rating(personal):
    bundle = _bundle(ratings=[(1, 5.0, _now())])
    expls = _explain(personal, bundle)
    assert expls[0]["primary"] == "Because you rated Deep Space 5★"


def test_rated_self_movie_references_own_rating(personal):
    bundle = _bundle(ratings=[(2, 4.0, _now())])
    expls = _explain(personal, bundle, results=[{"id": 2, "title": "Mars Survival"}])
    assert expls[0]["primary"] == "Because you rated Mars Survival 4★"


def test_strong_genre_preference_gives_genre_explanation(personal):
    bundle = _bundle(liked=[(1, _now()), (3, _now())], liked_ids=[1, 3])
    expls = _explain(personal, bundle, results=[{"id": 5, "title": "Alien Contact"}])
    combined = _texts(expls)[0]
    assert "Because you often watch Science Fiction" in combined


def test_watchlist_similarity_explanation(personal):
    bundle = _bundle(watchlist=[(1, _now())], watchlist_ids=[1])
    expls = _explain(personal, bundle)
    assert expls[0]["primary"] == "Because you've added similar movies to your watchlist"


def test_watchlisted_candidate_itself_explanation(personal):
    bundle = _bundle(watchlist=[(2, _now())], watchlist_ids=[2])
    expls = _explain(personal, bundle, results=[{"id": 2, "title": "Mars Survival"}])
    assert expls[0]["primary"] == "Because you added Mars Survival to your watchlist"


def test_recent_viewing_preference_explanation(personal):
    bundle = _bundle(views=[(1, _now(3))], viewed_ids=[1])
    expls = _explain(personal, bundle)
    assert expls[0]["primary"] == "Because you've been watching more Science Fiction lately"


def test_old_views_do_not_claim_lately(personal):
    bundle = _bundle(views=[(1, _now(60))], viewed_ids=[1])
    expls = _explain(personal, bundle)
    assert expls[0]["primary"] == "Recommended based on your preferences"


def test_recent_search_hint_and_no_raw_query_exposure(personal):
    bundle = _bundle(
        views=[(1, _now(2))], viewed_ids=[1],
        searches=[("a sci-fi thriller with a twist", _now(1))],
    )
    expls = _explain(personal, bundle, results=[{"id": 5, "title": "Alien Contact"}])
    text = _texts(expls)[0]
    assert "Because you recently searched for Science Fiction" in text
    assert "sci-fi thriller with a twist" not in text
    assert "twist" not in text


def test_multiple_reasons_capped_and_priority_respected(personal):
    bundle = _bundle(
        liked=[(1, _now())], liked_ids=[1],
        ratings=[(2, 5.0, _now())],
        watchlist=[(4, _now())], watchlist_ids=[4],
        views=[(1, _now())], viewed_ids=[1],
    )
    expls = _explain(personal, bundle, results=[{"id": 2, "title": "Mars Survival"}])
    e = expls[0]
    # exactly two reasons max, highest-priority genuine ones win - a strong
    # liked-similarity is the primary, a direct 5-star self-rating is secondary,
    # and lower-priority genre/viewing categories must not displace either
    assert e["primary"] == "Because you liked Deep Space"
    assert e["secondary"] == "Because you rated Mars Survival 5★"


def test_no_behavioral_signal_falls_back(personal):
    empty = _bundle()
    assert _explain(personal, empty, mode="personalized")[0]["primary"] == \
        "Recommended based on your preferences"
    assert _explain(personal, empty, mode="trending")[0]["primary"] == "Popular right now"


def test_negative_feedback_does_not_produce_positive_claim(personal):
    bundle = _bundle(feedback=[(3, "down", _now())], downvoted={3})
    expls = _explain(personal, bundle, results=[{"id": 2, "title": "Mars Survival"}])
    text = _texts(expls)[0]
    assert text == "Recommended based on your preferences"
    for forbidden in ("liked", "rated", "watchlist", "enjoyed", "searched", "watching"):
        assert forbidden.lower() not in text.lower()


def test_explanation_is_deterministic_across_order(personal):
    bundle_a = _bundle(liked=[(1, _now()), (3, _now())], liked_ids=[1, 3])
    bundle_b = _bundle(liked=[(3, _now()), (1, _now())], liked_ids=[3, 1])
    ra = _explain(personal, bundle_a, results=[{"id": 5, "title": "Alien Contact"}])
    rb = _explain(personal, bundle_b, results=[{"id": 5, "title": "Alien Contact"}])
    rc = _explain(personal, bundle_a, results=[{"id": 5, "title": "Alien Contact"}])
    assert ra == rc == rb


def test_existing_response_fields_remain_intact(personal):
    bundle = _bundle(liked=[(1, _now())], liked_ids=[1])
    results = [{
        "id": 2, "title": "Mars Survival", "match_percentage": 88,
        "reasons": ["some existing reason"], "confidence": {"label": "High", "score": 0.9},
    }]
    before = set(results[0].keys())
    # the service attaches the returned explanations onto the same result dicts
    expls = _explain(personal, bundle, results=results)
    for item, explanation in zip(results, expls):
        item["explanation"] = explanation
    after = set(results[0].keys())
    assert before <= after
    assert "explanation" in after
    assert results[0]["explanation"]["primary"] == "Because you liked Deep Space"
    assert results[0]["match_percentage"] == 88
    assert results[0]["reasons"] == ["some existing reason"]


def test_explanation_adds_no_database_queries():
    """Explanations must run purely on the already-batched bundle + ranking
    context: seeding creates the bundle inside one counted call, and the
    explanation layer afterwards must not execute a single SELECT."""
    from app.db.session import SessionLocal, engine
    from app.services import user_service
    from app.db import models

    session_id = f"pytest-explain-batch-{uuid4()}"
    db = SessionLocal()

    try:
        user_service.get_or_create_user(db, session_id)
        user_service.record_interaction(db, session_id, 1, "liked")

        count = {"n": 0}

        def count_selects(conn, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                count["n"] += 1

        # Bundle load is the ONE allowed set of queries (batched, ~1 per type).
        event.listen(engine, "before_cursor_execute", count_selects)
        bundle = user_service.get_signal_bundle(db, session_id)
        baseline = count["n"]

        provider = PersonalizationEngine(ContentSimilarityEngine(make_df()))
        _explain(provider, bundle)
        assert count["n"] == baseline  # explanation added zero queries

        event.remove(engine, "before_cursor_execute", count_selects)
    finally:
        user = db.query(models.User).filter(models.User.session_id == session_id).first()
        if user:
            db.query(models.UserInteraction).filter(
                models.UserInteraction.user_id == user.id
            ).delete(synchronize_session=False)
            db.query(models.User).filter(models.User.id == user.id).delete(
                synchronize_session=False
            )
        db.commit()
        db.close()