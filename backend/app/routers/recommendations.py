from fastapi import APIRouter, HTTPException, Depends
import asyncio
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.services.recommendation_service import recommendation_service
from app.services.review_fetcher import fetch_reviews
from app.services.tmdb_client import tmdb_client, TMDBError
from app.services.sentiment_cache import sentiment_cache
from app.services.data_prep import tmdb_movie_to_row
from app.services import user_service
from app.db.session import get_db
from app.config import settings
from ml_engine.explainer import explain_content_match, to_match_percentage
from ml_engine.aspect_sentiment import analyze_overall, analyze_aspects
from ml_engine.sentiment_model import SentimentModelNotTrainedError
from ml_engine.hybrid_ranker import compute_hybrid_score, compute_confidence, recency_score
from ml_engine.recommendation_explainer import build_recommendation_explanations

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
    Personalized recommendations built from the user's behavioral signals
    (likes, ratings, watchlist, views, searches, recommendation feedback).
    Falls back to a clearly-labeled "trending" list for cold-start users
    instead of pretending to personalize with no signal.
    """
    signals = user_service.get_signal_bundle(db, session_id)
    downvoted = signals["downvoted"]

    if not user_service.has_personalization_signal(signals):
        trending = recommendation_service.get_trending_fallback(top_n=top_n)
        trending = [m for m in trending if m["id"] not in downvoted]
        return {"mode": "trending", "reason": "Not enough preference data yet - showing trending titles.", "results": trending}

    personalized = recommendation_service.rank_personalized_candidates(signals, top_n=top_n)
    if not personalized:
        trending = recommendation_service.get_trending_fallback(top_n=top_n)
        trending = [m for m in trending if m["id"] not in downvoted]
        return {"mode": "trending", "reason": "Signals don't map to movies in the current dataset - showing trending titles.", "results": trending}

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

    preference_scores: dict[int, float] = {}
    behavior_boosts: dict[int, float] = {}
    signals: dict | None = None
    if session_id:
        signals = user_service.get_signal_bundle(db, session_id)
        liked_ids = signals["liked_ids"]
        downvoted = signals["downvoted"]
        liked_set = set(liked_ids)
        candidates = [c for c in candidates if c["id"] not in downvoted and c["id"] not in liked_set]
        preference_scores = recommendation_service._personalization.behavior_scores(signals)
        behavior_boosts = recommendation_service._personalization.movie_behavior_boosts(signals)

    if not candidates:
        return []

    rows = {c["id"]: recommendation_service.get_movie_row(c["id"]) for c in candidates}
    popularity_max = max((rows[c["id"]].get("popularity") or 0 for c in candidates), default=1)

    # Sentiment is computed once per movie and reused for the TTL: cache hits
    # skip both the TMDB review fetch and the model prediction. Cached payloads
    # carry an internal _review_count so the confidence estimate stays exact.
    def sentiment_key(cid: int) -> tuple:
        return sentiment_cache.build_key(cid, min_reviews=settings.min_reviews_for_sentiment)

    payload_by_id: dict[int, dict] = {}
    to_fetch: list[int] = []
    for c in candidates:
        cid = int(c["id"])
        cached = sentiment_cache.get(sentiment_key(cid))
        if cached is not None:
            payload_by_id[cid] = cached
        else:
            to_fetch.append(cid)

    if to_fetch:
        fetched_lists = await asyncio.gather(*(fetch_reviews(cid) for cid in to_fetch))
        for cid, reviews in zip(to_fetch, fetched_lists):
            if len(reviews) >= settings.min_reviews_for_sentiment:
                try:
                    overall = analyze_overall(reviews)
                    aspects = analyze_aspects(reviews, min_sentences=settings.min_reviews_for_sentiment)
                except SentimentModelNotTrainedError:
                    payload_by_id[cid] = {"movie_id": cid, "_review_count": len(reviews)}
                else:
                    payload = {
                        "movie_id": cid,
                        "overall": overall,
                        "aspects": aspects,
                        "_review_count": len(reviews),
                    }
                    sentiment_cache.set(sentiment_key(cid), payload)
                    payload_by_id[cid] = payload
            else:
                payload = {
                    "status": "insufficient_data",
                    "review_count": len(reviews),
                    "message": (
                        f"Only {len(reviews)} review(s) found - need at least "
                        f"{settings.min_reviews_for_sentiment} for a reliable score."
                    ),
                    "_review_count": len(reviews),
                }
                sentiment_cache.set(sentiment_key(cid), payload)
                payload_by_id[cid] = payload

    ranked = []
    for c in candidates:
        cid = int(c["id"])
        row = rows[cid]
        payload = payload_by_id[cid]
        overall = payload.get("overall")
        sentiment_pct = overall.get("positive_pct") if overall else None
        review_count = payload.get("_review_count", payload.get("review_count", 0))

        preference_match = preference_scores.get(cid) if preference_scores else None
        recency = recency_score(row.get("release_date"))
        score = compute_hybrid_score(
            content_similarity=c["similarity_score"],
            sentiment_positive_pct=sentiment_pct,
            rating=row.get("vote_average") or 0,
            popularity=row.get("popularity") or 0,
            popularity_max_in_batch=popularity_max,
            preference_match=preference_match,
            recency=recency,
            feedback_boost=behavior_boosts.get(cid, 0.0),
        )
        confidence = compute_confidence(
            content_similarity=c["similarity_score"],
            has_sentiment=sentiment_pct is not None,
            review_count=review_count,
            popularity=row.get("popularity") or 0,
            popularity_max_in_batch=popularity_max,
        )

        reasons = explain_content_match(c)
        if preference_match is not None and preference_match >= 0.65:
            reasons.append("Strong match with your taste profile")
        if (row.get("vote_average") or 0) >= 7.5:
            reasons.append("Highly rated by audiences")
        if recency >= 0.8:
            reasons.append("Relatively recent release")
        if sentiment_pct is not None:
            reasons.append(f"{sentiment_pct}% positive audience sentiment")
        else:
            reasons.append("Audience sentiment: insufficient review data")

        ranked.append(
            {
                "id": cid,
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
    diversified = recommendation_service.diversify(ranked, top_n=top_n)

    if signals is not None:
        weights = recommendation_service._personalization.aggregate_signal_weights(signals)
        explained = build_recommendation_explanations(
            diversified,
            recommendation_service._personalization,
            signals,
            weights=weights,
            preference_scores=preference_scores,
            behavior_boosts=behavior_boosts,
            mode="hybrid",
        )
        for item, explanation in zip(diversified, explained):
            item["explanation"] = explanation

    return diversified


@router.post("/feedback")
def submit_recommendation_feedback(payload: FeedbackIn, db: Session = Depends(get_db)):
    try:
        user_service.record_recommendation_feedback(
            db, payload.session_id, payload.source_movie_id, payload.recommended_movie_id, payload.feedback
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok"}
