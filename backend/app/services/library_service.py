"""Persistence helpers for the Phase 2 dynamic movie library."""
from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import LibraryMovie


LIST_FIELDS = ("genres", "cast", "director", "keywords")


def _as_csv(value) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return ", ".join(str(v) for v in value if v)


def row_to_db_values(row: dict) -> dict:
    values = dict(row)
    for field in LIST_FIELDS:
        values[field] = _as_csv(values.get(field))
    return {
        "tmdb_id": int(values["id"]),
        "title": values.get("title", ""),
        "overview": values.get("overview", ""),
        "genres": values.get("genres", ""),
        "cast": values.get("cast", ""),
        "director": values.get("director", ""),
        "keywords": values.get("keywords", ""),
        "poster_path": values.get("poster_path"),
        "backdrop_path": values.get("backdrop_path"),
        "release_date": values.get("release_date", ""),
        "vote_average": float(values.get("vote_average") or 0),
        "popularity": float(values.get("popularity") or 0),
    }


def library_movie_to_row(movie: LibraryMovie) -> dict:
    return {
        "id": movie.tmdb_id,
        "title": movie.title,
        "overview": movie.overview or "",
        "genres": movie.genres or "",
        "cast": movie.cast or "",
        "director": movie.director or "",
        "keywords": movie.keywords or "",
        "poster_path": movie.poster_path,
        "backdrop_path": movie.backdrop_path,
        "release_date": movie.release_date or "",
        "vote_average": movie.vote_average or 0,
        "popularity": movie.popularity or 0,
        "source": "tmdb",
    }


def upsert_library_movie(db: Session, row: dict) -> tuple[LibraryMovie, bool]:
    """Insert or refresh one TMDB movie. Returns (movie, was_new)."""
    values = row_to_db_values(row)
    movie = db.get(LibraryMovie, values["tmdb_id"])
    was_new = movie is None

    if movie is None:
        movie = LibraryMovie(**values)
        db.add(movie)
    else:
        for key, value in values.items():
            setattr(movie, key, value)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        # A concurrent request may have inserted the same TMDB id between
        # our lookup and commit. Refresh it and apply the current metadata.
        movie = db.get(LibraryMovie, values["tmdb_id"])
        if movie is None:
            raise
        for key, value in values.items():
            setattr(movie, key, value)
        db.commit()
        was_new = False

    db.refresh(movie)
    return movie, was_new
