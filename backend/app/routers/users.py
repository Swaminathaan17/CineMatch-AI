from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services import user_service

router = APIRouter()


class InteractionIn(BaseModel):
    session_id: str
    movie_id: int
    interaction_type: str  # 'liked' | 'viewed' | 'searched'


class GenresIn(BaseModel):
    session_id: str
    genres: list[str]


class WatchlistIn(BaseModel):
    session_id: str
    movie_id: int
    movie_title: str | None = None


@router.post("/interactions")
def log_interaction(payload: InteractionIn, db: Session = Depends(get_db)):
    user_service.record_interaction(
        db, payload.session_id, payload.movie_id, payload.interaction_type
    )
    return {"status": "ok"}


@router.post("/favorite-genres")
def set_favorite_genres(payload: GenresIn, db: Session = Depends(get_db)):
    user_service.set_favorite_genres(db, payload.session_id, payload.genres)
    return {"status": "ok"}


@router.get("/preferences")
def get_preferences(session_id: str, db: Session = Depends(get_db)):
    return {
        "liked_movie_ids": user_service.get_liked_movie_ids(db, session_id),
        "favorite_genres": user_service.get_favorite_genres(db, session_id),
    }


@router.post("/watchlist")
def add_to_watchlist(payload: WatchlistIn, db: Session = Depends(get_db)):
    user_service.add_to_watchlist(db, payload.session_id, payload.movie_id, payload.movie_title)
    return {"status": "ok"}


@router.delete("/watchlist/{movie_id}")
def remove_from_watchlist(movie_id: int, session_id: str, db: Session = Depends(get_db)):
    user_service.remove_from_watchlist(db, session_id, movie_id)
    return {"status": "ok"}


@router.get("/watchlist")
def get_watchlist(session_id: str, db: Session = Depends(get_db)):
    return {"results": user_service.get_watchlist(db, session_id)}
