from fastapi import APIRouter, HTTPException

from app.services.review_fetcher import fetch_reviews
from app.services.sentiment_cache import sentiment_cache, as_public_payload
from app.config import settings
from ml_engine.aspect_sentiment import analyze_overall, analyze_aspects
from ml_engine.sentiment_model import SentimentModelNotTrainedError

router = APIRouter()


@router.get("/{movie_id}")
async def get_sentiment(movie_id: int):
    key = sentiment_cache.build_key(
        movie_id,
        min_reviews=settings.min_reviews_for_sentiment,
    )
    cached = sentiment_cache.get(key)
    if cached is not None:
        return as_public_payload(cached)

    reviews = await fetch_reviews(movie_id)

    if len(reviews) < settings.min_reviews_for_sentiment:
        result = {
            "status": "insufficient_data",
            "review_count": len(reviews),
            "message": (
                f"Only {len(reviews)} review(s) found - need at least "
                f"{settings.min_reviews_for_sentiment} for a reliable score."
            ),
            "_review_count": len(reviews),
        }
        sentiment_cache.set(key, result)
        return as_public_payload(result)

    try:
        overall = analyze_overall(reviews)
        aspects = analyze_aspects(reviews, min_sentences=settings.min_reviews_for_sentiment)
    except SentimentModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    result = {
        "movie_id": movie_id,
        "overall": overall,
        "aspects": aspects,
        "_review_count": len(reviews),
    }
    sentiment_cache.set(key, result)
    return as_public_payload(result)
