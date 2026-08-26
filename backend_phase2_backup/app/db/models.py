"""SQLite persistence models for users, feedback, and the growing movie library."""
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Text
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
    vote_average = Column(Float, default=0.0)
    popularity = Column(Float, default=0.0)
    added_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
