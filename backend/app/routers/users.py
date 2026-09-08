from fastapi import APIRouter, Depends, HTTPException
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


class SearchIn(BaseModel):
    session_id: str
    query: str
    result_source: str | None = None
    result_count: int | None = None


class RatingIn(BaseModel):
    session_id: str
    movie_id: int
    rating: float


@router.post("/interactions")
def log_interaction(payload: InteractionIn, db: Session = Depends(get_db)):
    user_service.record_interaction(
        db, payload.session_id, payload.movie_id, payload.interaction_type
    )
    return {"status": "ok"}


@router.get("/interactions")
def get_interactions(session_id: str, db: Session = Depends(get_db)):
    return {"results": user_service.get_interactions(db, session_id)}


@router.post("/searches")
def log_search(payload: SearchIn, db: Session = Depends(get_db)):
    user_service.record_search_query(
        db,
        payload.session_id,
        payload.query,
        result_source=payload.result_source,
        result_count=payload.result_count,
    )
    return {"status": "ok"}


@router.get("/searches")
def get_searches(session_id: str, db: Session = Depends(get_db)):
    return {"results": user_service.get_recent_searches(db, session_id)}


@router.post("/ratings")
def set_movie_rating(payload: RatingIn, db: Session = Depends(get_db)):
    try:
        user_service.rate_movie(db, payload.session_id, payload.movie_id, payload.rating)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok"}


@router.get("/ratings")
def get_movie_ratings(session_id: str, db: Session = Depends(get_db)):
    return {"ratings": user_service.get_ratings(db, session_id)}


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


@router.get("/watchlist/events")
def get_watchlist_events(session_id: str, db: Session = Depends(get_db)):
    return {"results": user_service.get_watchlist_events(db, session_id)}
