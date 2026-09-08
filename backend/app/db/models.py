"""SQLite persistence models for users, feedback, and the growing movie library."""
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=utcnow)

    interactions = relationship("UserInteraction", back_populates="user")
    genre_preferences = relationship("UserGenrePreference", back_populates="user")


class UserGenrePreference(Base):
    __tablename__ = "user_genre_preferences"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    genre = Column(String, nullable=False)
    weight = Column(Float, default=1.0)

    user = relationship("User", back_populates="genre_preferences")


class UserInteraction(Base):
    __tablename__ = "user_interactions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    movie_id = Column(Integer, nullable=False)
    interaction_type = Column(String, nullable=False)  # 'liked' | 'viewed' | 'searched'
    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="interactions")


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    movie_id = Column(Integer, nullable=False)
    movie_title = Column(String, nullable=True)
    added_at = Column(DateTime, default=utcnow)


class RecommendationFeedback(Base):
    __tablename__ = "recommendation_feedback"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    source_movie_id = Column(Integer, nullable=False)
    recommended_movie_id = Column(Integer, nullable=False)
    feedback = Column(String, nullable=False)  # 'up' | 'down'
    created_at = Column(DateTime, default=utcnow)


class MovieRating(Base):
    """Explicit 1-5 star rating a user gives a movie.

    One rating per (user, movie): re-rating overwrites the value and refreshes
    updated_at rather than stacking duplicate rows. created_at keeps the first
    rating time, updated_at the latest, so the future recommendation engine can
    weight by rating recency.
    """

    __tablename__ = "movie_ratings"
    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_movie_ratings_user_movie"),
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    movie_id = Column(Integer, nullable=False)
    rating = Column(Float, nullable=False)  # 1-5
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class UserSearchQuery(Base):
    """Search-question history used for personalization.

    Each distinct search is preserved as its own row. Repeating the same query
    within a short window refreshes the existing row's timestamp instead of
    inserting duplicate spam.
    """

    __tablename__ = "user_search_queries"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    query = Column(String, nullable=False)
    result_source = Column(String, nullable=True)  # 'local' | 'tmdb' | None
    result_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=utcnow)


class WatchlistEvent(Base):
    """Add/remove history for watchlist signals.

    The watchlist_items table stays the source of truth for current state;
    this table is an append-only log of what changed and when so the future
    recommendation engine can weigh watchlist behavior.
    """

    __tablename__ = "watchlist_events"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    movie_id = Column(Integer, nullable=False)
    movie_title = Column(String, nullable=True)
    action = Column(String, nullable=False)  # 'added' | 'removed'
    created_at = Column(DateTime, default=utcnow)


class LibraryMovie(Base):
    """Movies explicitly imported from TMDB into the persistent library.

    The base CSV remains the original catalogue. This table is the Phase 2
    extension layer, so upgrading the app never rewrites the original data.
    TMDB ids are globally unique in our movie namespace and are therefore used
    as the primary key as well as the natural lookup key.
    """

    __tablename__ = "library_movies"

    tmdb_id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    overview = Column(Text, nullable=True)
    genres = Column(Text, nullable=True)
    cast = Column(Text, nullable=True)
    director = Column(Text, nullable=True)
    keywords = Column(Text, nullable=True)
    poster_path = Column(String, nullable=True)
    backdrop_path = Column(String, nullable=True)
    release_date = Column(String, nullable=True)
    runtime = Column(Integer, nullable=True)  # minutes; None if TMDB has no value
    vote_average = Column(Float, default=0.0)
    popularity = Column(Float, default=0.0)
    added_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class MovieAsset(Base):
    """Enrichment cache for original-catalogue movie assets backfilled from TMDB.

    The source CSV stays authoritative and immutable; this table only fills
    asset gaps (poster/backdrop/release date/runtime) for catalogue movies so
    a backfill never rewrites or deletes anything else about a movie. Rows are
    keyed by the same TMDB id used across the movie namespace.
    """

    __tablename__ = "movie_assets"

    tmdb_id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    poster_path = Column(String, nullable=True)
    backdrop_path = Column(String, nullable=True)
    release_date = Column(String, nullable=True)
    runtime = Column(Integer, nullable=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
