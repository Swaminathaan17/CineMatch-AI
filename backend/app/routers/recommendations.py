from fastapi import APIRouter, HTTPException, Depends
import asyncio
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.services.recommendation_service import recommendation_service
from app.services.review_fetcher import fetch_reviews
from app.services.tmdb_client import tmdb_client, TMDBError
from app.services.data_prep import tmdb_movie_to_row
from app.services import user_service
from app.db.session import get_db
from app.config import settings
from ml_engine.explainer import explain_content_match, to_match_percentage
from ml_engine.aspect_sentiment import analyze_overall
from ml_engine.sentiment_model import SentimentModelNotTrainedError
from ml_engine.hybrid_ranker import compute_hybrid_score, compute_confidence, recency_score

router = APIRouter()


class FeedbackIn(BaseModel):
    session_id: str
    source_movie_id: int
    recommended_movie_id: int
    feedback: str  # 'up' | 'down'


@router.get("/personalized")
def get_personalized_recommendations(
    session_id: str, top_n: int = 10, db: Session = Depends(get_db)
):
    """
    Personalized recommendations built from the user's liked movies + favorite
    genres. Falls back to a clearly-labeled "trending" list for cold-start
    users instead of pretending to personalize with no signal.
    """
    liked_ids = user_service.get_liked_movie_ids(db, session_id)
    genres = user_service.get_favorite_genres(db, session_id)
    downvoted = user_service.get_downvoted_movie_ids(db, session_id)
    feedback_map = user_service.get_feedback_map(db, session_id)

    if not liked_ids and not genres:
        trending = recommendation_service.get_trending_fallback(top_n=top_n)
        trending = [m for m in trending if m["id"] not in downvoted]
        return {"mode": "trending", "reason": "Not enough preference data yet - showing trending titles.", "results": trending}

    personalized = recommendation_service.rank_personalized_candidates(
        liked_ids, genres, downvoted=downvoted, feedback_map=feedback_map, top_n=top_n
    )
    if not personalized:
        trending = recommendation_service.get_trending_fallback(top_n=top_n)
        trending = [m for m in trending if m["id"] not in downvoted]
        return {"mode": "trending", "reason": "Liked movies not found in current dataset - showing trending titles.", "results": trending}

    return {"mode": "personalized", "results": personalized}


@router.get("/tmdb/{movie_id}")
async def get_tmdb_recommendations(movie_id: int, top_n: int = 10):
    """Recommend movies from our local catalogue for a TMDB-only movie.

    The TMDB title is never inserted into the local dataset. Its metadata is
    converted to the same feature shape and used as a temporary query vector.
    """
    try:
        tmdb_data = await tmdb_client.get_movie(movie_id)
    except TMDBError as e:
        raise HTTPException(status_code=502, detail=str(e))

    external_row = tmdb_movie_to_row(tmdb_data)
    matches = recommendation_service.get_similar_to_external(external_row, top_n=top_n)

    return [
        {
            "id": m["id"],
            "title": m["title"],
            "match_percentage": to_match_percentage(m["similarity_score"]),
            "reasons": explain_content_match(m),
        }
        for m in matches
    ]


@router.get("/trending")
def get_trending_recommendations(top_n: int = 20):
    """Fast catalogue-wide fallback using rating + popularity."""
    return recommendation_service.get_trending_fallback(top_n=top_n)


@router.get("/{movie_id}")
def get_recommendations(movie_id: int, top_n: int = 10):
    """
    Content-based recommendations. Ranked purely on similarity - fast,
    no external calls needed. Use /recommendations/{movie_id}/hybrid for the
    full sentiment+rating+popularity-blended ranking.
    """
    try:
        matches = recommendation_service.get_similar(movie_id, top_n=top_n)
    except ValueError:
        raise HTTPException(status_code=404, detail="Movie not in local dataset")

    return [
        {
            "id": m["id"],
            "title": m["title"],
            "match_percentage": to_match_percentage(m["similarity_score"]),
            "reasons": explain_content_match(m),
        }
        for m in matches
    ]


@router.get("/{movie_id}/hybrid")
async def get_hybrid_recommendations(
    movie_id: int,
    top_n: int = 10,
    candidate_pool: int = 40,
    session_id: str | None = None,
    db: Session = Depends(get_db),
):
    """
    Full hybrid ranking: content similarity + sentiment + rating + popularity.
    Slower than the plain endpoint above since it fetches reviews per candidate -
    pulls a wider candidate_pool from content similarity first, then re-ranks
    that pool using the hybrid score, and returns the top_n after re-ranking.

    If session_id is provided, movies this user has previously downvoted as
    a recommendation are excluded from the candidate pool entirely.
    """
    try:
        candidates = recommendation_service.get_similar(movie_id, top_n=candidate_pool)
    except ValueError:
        raise HTTPException(status_code=404, detail="Movie not in local dataset")

    liked_ids: list[int] = []
    genres: list[str] = []
    feedback_map: dict[int, str] = {}
    if session_id:
        liked_ids = user_service.get_liked_movie_ids(db, session_id)
        genres = user_service.get_favorite_genres(db, session_id)
        downvoted = user_service.get_downvoted_movie_ids(db, session_id)
        feedback_map = user_service.get_feedback_map(db, session_id)
        candidates = [c for c in candidates if c["id"] not in downvoted and c["id"] not in set(liked_ids)]

    if not candidates:
        return []

    rows = {c["id"]: recommendation_service.get_movie_row(c["id"]) for c in candidates}
    popularity_max = max((rows[c["id"]].get("popularity") or 0 for c in candidates), default=1)
    negative_ids = [mid for mid, fb in feedback_map.items() if fb == "down"]
    preference_scores = (
        recommendation_service._personalization.preference_scores(liked_ids, genres, negative_ids)
        if session_id else {}
    )

    # Reviews are independent network calls. Fetch them concurrently so a
    # wider accuracy-oriented candidate pool does not multiply page latency.
    review_lists = await asyncio.gather(*(fetch_reviews(c["id"]) for c in candidates))

    ranked = []
    for c, reviews in zip(candidates, review_lists):
        row = rows[c["id"]]
        sentiment_pct = None
        if len(reviews) >= settings.min_reviews_for_sentiment:
            try:
                overall = analyze_overall(reviews)
                sentiment_pct = overall.get("positive_pct")
            except SentimentModelNotTrainedError:
                sentiment_pct = None  # degrade gracefully, don't fail the whole request

        preference_match = preference_scores.get(c["id"]) if preference_scores else None
        score = compute_hybrid_score(
            content_similarity=c["similarity_score"],
            sentiment_positive_pct=sentiment_pct,
            rating=row.get("vote_average") or 0,
            popularity=row.get("popularity") or 0,
            popularity_max_in_batch=popularity_max,
            preference_match=preference_match,
            recency=recency_score(row.get("release_date")),
            feedback_boost=0.03 if feedback_map.get(c["id"]) == "up" else 0.0,
        )
        confidence = compute_confidence(
            content_similarity=c["similarity_score"],
            has_sentiment=sentiment_pct is not None,
            review_count=len(reviews),
            popularity=row.get("popularity") or 0,
            popularity_max_in_batch=popularity_max,
        )

        reasons = explain_content_match(c)
        if preference_match is not None and preference_match >= 0.65:
            reasons.append("Strong match with your taste profile")
        if (row.get("vote_average") or 0) >= 7.5:
            reasons.append("Highly rated by audiences")
        if recency_score(row.get("release_date")) >= 0.8:
            reasons.append("Relatively recent release")
        if sentiment_pct is not None:
            reasons.append(f"{sentiment_pct}% positive audience sentiment")
        else:
            reasons.append("Audience sentiment: insufficient review data")

        ranked.append(
            {
                "id": c["id"],
                "title": c["title"],
                "match_percentage": min(99, round(score["final_score"] * 100)),
                "score_breakdown": score["components"],
                "confidence": confidence,
                "reasons": reasons,
            }
        )

    ranked.sort(key=lambda r: r["match_percentage"], reverse=True)
    for item in ranked:
        item["_raw_score"] = item["match_percentage"] / 100
    return recommendation_service.diversify(ranked, top_n=top_n)


@router.post("/feedback")
def submit_recommendation_feedback(payload: FeedbackIn, db: Session = Depends(get_db)):
    try:
        user_service.record_recommendation_feedback(
            db, payload.session_id, payload.source_movie_id, payload.recommended_movie_id, payload.feedback
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok"}
