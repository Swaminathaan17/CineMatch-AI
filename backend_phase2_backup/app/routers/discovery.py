from fastapi import APIRouter
from pydantic import BaseModel

from app.services.recommendation_service import recommendation_service

router = APIRouter()


class QueryIn(BaseModel):
    query: str


@router.post("/query")
def discover(payload: QueryIn, top_n: int = 10):
    result = recommendation_service.search_by_mood(payload.query, top_n=top_n)
    return {
        "query": payload.query,
        "mode": result["mode"],  # "semantic" or "fallback" - be honest about which
        "interpreted_genres": result["interpreted_genres"],
        "results": result["results"],
    }
