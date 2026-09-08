"""
Session-based users - no passwords/auth, just a session_id the frontend
generates once (e.g. a UUID in localStorage) and sends with requests. Enough
for a college project demo where the point is showing personalization works,
not building a full auth system.
"""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy.orm import Session

from app.db.models import (
    User,
    UserInteraction,
    UserGenrePreference,
    WatchlistItem,
    RecommendationFeedback,
    MovieRating,
    UserSearchQuery,
    WatchlistEvent,
    utcnow,
)

# Repeating the exact same search within this window refreshes the existing
# row's timestamp instead of inserting duplicate spam. Distinct searches are
# always preserved - this only guards against repeat-submit noise.
SEARCH_DEDUPE_WINDOW = timedelta(minutes=30)


def get_or_create_user(db: Session, session_id: str) -> User:
    user = db.query(User).filter(User.session_id == session_id).first()
    if user is None:
        user = User(session_id=session_id)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def record_interaction(db: Session, session_id: str, movie_id: int, interaction_type: str) -> None:
    """Record a movie interaction for a user.

    Semantics:
    - 'liked': one active like per (user, movie). Re-liking is idempotent and
      refreshes the original row's timestamp rather than inserting duplicates.
    - 'viewed'/'searched': append-only behavioral events - repeated views are
      preserved as separate rows so the rec engine can observe view frequency.
    """
    user = get_or_create_user(db, session_id)
    if interaction_type == "liked":
        existing = (
            db.query(UserInteraction)
            .filter(
                UserInteraction.user_id == user.id,
                UserInteraction.movie_id == movie_id,
                UserInteraction.interaction_type == "liked",
            )
            .first()
        )
        if existing:
            existing.created_at = utcnow()
            db.commit()
            return
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
    db.add(WatchlistEvent(user_id=user.id, movie_id=movie_id, movie_title=movie_title, action="added"))
    db.commit()


def remove_from_watchlist(db: Session, session_id: str, movie_id: int) -> None:
    user = get_or_create_user(db, session_id)
    item = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == user.id, WatchlistItem.movie_id == movie_id)
        .first()
    )
    if item is None:
        return  # idempotent - removing what isn't there is a no-op
    db.delete(item)
    db.add(WatchlistEvent(user_id=user.id, movie_id=movie_id, movie_title=item.movie_title, action="removed"))
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


def rate_movie(db: Session, session_id: str, movie_id: int, rating: float) -> None:
    """Store a user's 1-5 star rating for a movie.

    One rating per (user, movie): re-rating overwrites the previous value and
    refreshes updated_at via the model's onupdate. The original created_at is
    preserved.
    """
    if not (1 <= rating <= 5):
        raise ValueError("rating must be between 1 and 5")
    user = get_or_create_user(db, session_id)
    existing = (
        db.query(MovieRating)
        .filter(MovieRating.user_id == user.id, MovieRating.movie_id == movie_id)
        .first()
    )
    if existing:
        existing.rating = rating
    else:
        db.add(MovieRating(user_id=user.id, movie_id=movie_id, rating=rating))
    db.commit()


def get_ratings(db: Session, session_id: str) -> dict[int, float]:
    """Map of movie_id -> user's rating (latest value only)."""
    user = get_or_create_user(db, session_id)
    rows = (
        db.query(MovieRating.movie_id, MovieRating.rating)
        .filter(MovieRating.user_id == user.id)
        .all()
    )
    return {movie_id: rating for movie_id, rating in rows}


def record_search_query(
    db: Session,
    session_id: str,
    query: str,
    result_source: str | None = None,
    result_count: int | None = None,
) -> None:
    """Log a search query for personalization.

    Distinct queries are always preserved. Repeating the exact same query
    within SEARCH_DEDUPE_WINDOW refreshes the existing row's timestamp instead
    of stacking duplicate rows, so spinner spam never floods the history.
    """
    cleaned = query.strip()[:300]
    if not cleaned:
        return
    user = get_or_create_user(db, session_id)
    cutoff = utcnow() - SEARCH_DEDUPE_WINDOW
    existing = (
        db.query(UserSearchQuery)
        .filter(
            UserSearchQuery.user_id == user.id,
            UserSearchQuery.query == cleaned,
            UserSearchQuery.created_at >= cutoff,
        )
        .order_by(UserSearchQuery.created_at.desc())
        .first()
    )
    if existing:
        existing.created_at = utcnow()
    else:
        db.add(
            UserSearchQuery(
                user_id=user.id,
                query=cleaned,
                result_source=result_source,
                result_count=result_count,
            )
        )
    db.commit()


def get_recent_searches(db: Session, session_id: str) -> list[dict]:
    """Raw search history, newest first, for future personalization."""
    user = get_or_create_user(db, session_id)
    rows = (
        db.query(UserSearchQuery)
        .filter(UserSearchQuery.user_id == user.id)
        .order_by(UserSearchQuery.created_at.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "id": r.id,
            "query": r.query,
            "result_source": r.result_source,
            "result_count": r.result_count,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


def get_interactions(db: Session, session_id: str) -> list[dict]:
    """All of a user's movie interactions, newest first."""
    user = get_or_create_user(db, session_id)
    rows = (
        db.query(UserInteraction)
        .filter(UserInteraction.user_id == user.id)
        .order_by(UserInteraction.created_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "movie_id": r.movie_id,
            "interaction_type": r.interaction_type,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


def get_viewed_movie_ids(db: Session, session_id: str) -> set[int]:
    """Distinct movies the user has ever viewed - a weak positive signal the
    future recommendation engine can use alongside likes/ratings."""
    user = get_or_create_user(db, session_id)
    rows = (
        db.query(UserInteraction.movie_id)
        .filter(UserInteraction.user_id == user.id, UserInteraction.interaction_type == "viewed")
        .all()
    )
    return {r[0] for r in rows}


def get_watchlist_events(db: Session, session_id: str) -> list[dict]:
    """Add/remove watchlist history, newest first."""
    user = get_or_create_user(db, session_id)
    rows = (
        db.query(WatchlistEvent)
        .filter(WatchlistEvent.user_id == user.id)
        .order_by(WatchlistEvent.created_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "movie_id": r.movie_id,
            "movie_title": r.movie_title,
            "action": r.action,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


def get_signal_bundle(db: Session, session_id: str) -> dict:
    """Load every tracked user signal in a small, fixed number of batched
    queries (one per signal type - never once per candidate movie).

    The bundle is the input to the Phase 2.2 behavioral personalization engine
    and carries timestamps so recency weighting can be applied downstream.
    """
    user = get_or_create_user(db, session_id)

    liked = (
        db.query(UserInteraction.movie_id, UserInteraction.created_at)
        .filter(UserInteraction.user_id == user.id, UserInteraction.interaction_type == "liked")
        .all()
    )
    viewed = (
        db.query(UserInteraction.movie_id, UserInteraction.created_at)
        .filter(UserInteraction.user_id == user.id, UserInteraction.interaction_type == "viewed")
        .all()
    )
    ratings = (
        db.query(MovieRating.movie_id, MovieRating.rating, MovieRating.updated_at)
        .filter(MovieRating.user_id == user.id)
        .all()
    )
    watchlist = (
        db.query(WatchlistItem.movie_id, WatchlistItem.added_at)
        .filter(WatchlistItem.user_id == user.id)
        .all()
    )
    genres = (
        db.query(UserGenrePreference.genre)
        .filter(UserGenrePreference.user_id == user.id)
        .all()
    )
    searches = (
        db.query(UserSearchQuery.query, UserSearchQuery.created_at)
        .filter(UserSearchQuery.user_id == user.id)
        .order_by(UserSearchQuery.created_at.desc())
        .limit(10)
        .all()
    )
    feedback = (
        db.query(
            RecommendationFeedback.recommended_movie_id,
            RecommendationFeedback.feedback,
            RecommendationFeedback.created_at,
        )
        .filter(RecommendationFeedback.user_id == user.id)
        .all()
    )

    return {
        "liked": [(r[0], r[1]) for r in liked],
        "liked_ids": [r[0] for r in liked],
        "views": [(r[0], r[1]) for r in viewed],
        "viewed_ids": [r[0] for r in viewed],
        "ratings": [(r[0], r[1], r[2]) for r in ratings],
        "watchlist": [(r[0], r[1]) for r in watchlist],
        "watchlist_ids": [r[0] for r in watchlist],
        "favorite_genres": [r[0] for r in genres],
        "searches": [(r[0], r[1]) for r in searches],
        "feedback": [(r[0], r[1], r[2]) for r in feedback],
        "downvoted": {r[0] for r in feedback if r[1] == "down"},
    }


def has_personalization_signal(bundle: dict) -> bool:
    """Whether the bundle contains any signal at all (used for cold-start
    fallback to trending)."""
    return bool(
        bundle["liked"]
        or bundle["ratings"]
        or bundle["watchlist"]
        or bundle["views"]
        or bundle["feedback"]
    )
