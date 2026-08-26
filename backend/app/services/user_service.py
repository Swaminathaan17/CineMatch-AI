"""
Session-based users - no passwords/auth, just a session_id the frontend
generates once (e.g. a UUID in localStorage) and sends with requests. Enough
for a college project demo where the point is showing personalization works,
not building a full auth system.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import User, UserInteraction, UserGenrePreference, WatchlistItem, RecommendationFeedback


def get_or_create_user(db: Session, session_id: str) -> User:
    user = db.query(User).filter(User.session_id == session_id).first()
    if user is None:
        user = User(session_id=session_id)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def record_interaction(db: Session, session_id: str, movie_id: int, interaction_type: str) -> None:
    user = get_or_create_user(db, session_id)
    db.add(
        UserInteraction(user_id=user.id, movie_id=movie_id, interaction_type=interaction_type)
    )
    db.commit()


def set_favorite_genres(db: Session, session_id: str, genres: list[str]) -> None:
    user = get_or_create_user(db, session_id)
    db.query(UserGenrePreference).filter(UserGenrePreference.user_id == user.id).delete()
    for genre in genres:
        db.add(UserGenrePreference(user_id=user.id, genre=genre, weight=1.0))
    db.commit()


def get_liked_movie_ids(db: Session, session_id: str) -> list[int]:
    user = get_or_create_user(db, session_id)
    rows = (
        db.query(UserInteraction.movie_id)
        .filter(UserInteraction.user_id == user.id, UserInteraction.interaction_type == "liked")
        .all()
    )
    return [r[0] for r in rows]


def get_favorite_genres(db: Session, session_id: str) -> list[str]:
    user = get_or_create_user(db, session_id)
    rows = (
        db.query(UserGenrePreference.genre)
        .filter(UserGenrePreference.user_id == user.id)
        .all()
    )
    return [r[0] for r in rows]


def add_to_watchlist(db: Session, session_id: str, movie_id: int, movie_title: str | None) -> None:
    user = get_or_create_user(db, session_id)
    exists = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == user.id, WatchlistItem.movie_id == movie_id)
        .first()
    )
    if exists:
        return  # idempotent - adding twice isn't an error
    db.add(WatchlistItem(user_id=user.id, movie_id=movie_id, movie_title=movie_title))
    db.commit()


def remove_from_watchlist(db: Session, session_id: str, movie_id: int) -> None:
    user = get_or_create_user(db, session_id)
    db.query(WatchlistItem).filter(
        WatchlistItem.user_id == user.id, WatchlistItem.movie_id == movie_id
    ).delete()
    db.commit()


def get_watchlist(db: Session, session_id: str) -> list[dict]:
    user = get_or_create_user(db, session_id)
    rows = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == user.id)
        .order_by(WatchlistItem.added_at.desc())
        .all()
    )
    return [{"id": r.movie_id, "title": r.movie_title, "added_at": r.added_at.isoformat()} for r in rows]


def is_in_watchlist(db: Session, session_id: str, movie_id: int) -> bool:
    user = get_or_create_user(db, session_id)
    return (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == user.id, WatchlistItem.movie_id == movie_id)
        .first()
        is not None
    )


def record_recommendation_feedback(
    db: Session, session_id: str, source_movie_id: int, recommended_movie_id: int, feedback: str
) -> None:
    if feedback not in ("up", "down"):
        raise ValueError("feedback must be 'up' or 'down'")
    user = get_or_create_user(db, session_id)
    # one feedback per (user, source, recommended) triple - upsert rather
    # than accumulate duplicate rows if someone taps the button twice
    existing = (
        db.query(RecommendationFeedback)
        .filter(
            RecommendationFeedback.user_id == user.id,
            RecommendationFeedback.source_movie_id == source_movie_id,
            RecommendationFeedback.recommended_movie_id == recommended_movie_id,
        )
        .first()
    )
    if existing:
        existing.feedback = feedback
    else:
        db.add(
            RecommendationFeedback(
                user_id=user.id,
                source_movie_id=source_movie_id,
                recommended_movie_id=recommended_movie_id,
                feedback=feedback,
            )
        )
    db.commit()


def get_downvoted_movie_ids(db: Session, session_id: str) -> set[int]:
    """Movies this user has explicitly downvoted as recommendations, across
    any source movie - used to exclude them from future recommendation
    lists for this user."""
    user = get_or_create_user(db, session_id)
    rows = (
        db.query(RecommendationFeedback.recommended_movie_id)
        .filter(RecommendationFeedback.user_id == user.id, RecommendationFeedback.feedback == "down")
        .all()
    )
    return {r[0] for r in rows}


def get_feedback_map(db: Session, session_id: str) -> dict[int, str]:
    """Latest recommendation feedback by movie for lightweight ranking nudges."""
    user = get_or_create_user(db, session_id)
    rows = (
        db.query(RecommendationFeedback.recommended_movie_id, RecommendationFeedback.feedback)
        .filter(RecommendationFeedback.user_id == user.id)
        .all()
    )
    return {movie_id: feedback for movie_id, feedback in rows}
