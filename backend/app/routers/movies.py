from fastapi import APIRouter, HTTPException, Depends

from app.services.recommendation_service import recommendation_service
from app.services.tmdb_client import tmdb_client, TMDBError
from app.services.data_prep import tmdb_movie_to_row
from app.config import settings
from app.db.session import get_db
from app.services.library_service import upsert_library_movie
from sqlalchemy.orm import Session

router = APIRouter()


@router.get("/")
def list_movies():
    """Movies currently in our local dataset (used for similarity)."""
    return recommendation_service.list_movies()


# NOTE: literal-path routes (/search, /search/tmdb) MUST be registered
# before the /{movie_id} catch-all below. FastAPI/Starlette match routes in
# registration order, and a literal path segment like "search" would
# otherwise be swallowed by {movie_id}'s pattern first, causing a 422
# (int conversion failure) instead of ever reaching these handlers. This
# bug bit us once already in the recommendations router (Week 3) - same
# fix applies here.

@router.get("/search")
async def search_movies(q: str, top_n: int = 8):
    """
    Search local dataset first (instant, no API cost). If nothing matches
    AND a TMDB key is configured, fall back to a live TMDB search - so a
    movie outside the original static catalogue (or anything else missing
    locally) can still be found, without paying the TMDB round-trip for
    every search that the local data already answers.
    """
    if not q or not q.strip():
        return {"found": False, "source": None, "results": [], "message": "Enter a movie title to search."}

    local_results = recommendation_service.search_local(q, top_n=top_n)
    if local_results:
        return {"found": True, "source": "local", "results": local_results}

    if not settings.tmdb_api_key:
        return {
            "found": False,
            "source": None,
            "results": [],
            "message": (
                f"'{q}' isn't in our current library, "
                "and no TMDB key is configured for live search. Try AI Discovery "
                "to describe what you're looking for instead."
            ),
        }

    try:
        tmdb_data = await tmdb_client.search_movies(q)
        tmdb_results = [
            {
                "id": r["id"],
                "title": r.get("title", ""),
                "source": "tmdb",
                "poster_path": r.get("poster_path"),
                "release_date": r.get("release_date", ""),
            }
            for r in tmdb_data.get("results", [])[:top_n]
        ]
    except TMDBError:
        tmdb_results = []

    if not tmdb_results:
        return {
            "found": False,
            "source": None,
            "results": [],
            "message": f"'{q}' wasn't found locally or on TMDB. Try AI Discovery instead.",
        }

    return {
        "found": True,
        "source": "tmdb",
        "results": tmdb_results,
        "note": (
            "Found on TMDB. Open it to preview the movie, or add it to your "
            "persistent library to make it part of the recommendation engine."
        ),
    }


@router.get("/search/tmdb")
async def search_tmdb(q: str):
    """Live search against TMDB directly - for callers that want raw TMDB
    results specifically rather than the local-first /search behavior above."""
    try:
        return await tmdb_client.search_movies(q)
    except TMDBError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/{movie_id}/library")
async def add_tmdb_movie_to_library(movie_id: int, db: Session = Depends(get_db)):
    """Persist a TMDB movie in the growing local library and rebuild the index.

    Adding is idempotent: importing the same TMDB movie again refreshes its
    metadata instead of creating a duplicate. Movies already present in the
    original CSV are never duplicated into the extension table.
    """
    existing = recommendation_service.get_movie_row(movie_id)
    if existing is not None:
        if existing.get("source") == "local":
            return {"status": "already_in_library", "source": "local", "movie": existing}

        # A previously imported TMDB movie is already persistent. Refresh its
        # metadata from TMDB so title/poster/credits stay current.
        try:
            tmdb_data = await tmdb_client.get_movie(movie_id)
            row = tmdb_movie_to_row(tmdb_data)
            movie, _ = upsert_library_movie(db, row)
            recommendation_service.refresh()
            return {
                "status": "updated",
                "source": "tmdb",
                "movie": recommendation_service.get_movie_row(movie.tmdb_id),
            }
        except TMDBError as e:
            raise HTTPException(status_code=502, detail=str(e))

    try:
        tmdb_data = await tmdb_client.get_movie(movie_id)
    except TMDBError as e:
        raise HTTPException(status_code=502, detail=str(e))

    row = tmdb_movie_to_row(tmdb_data)
    movie, was_new = upsert_library_movie(db, row)
    recommendation_service.refresh()

    return {
        "status": "added" if was_new else "updated",
        "source": "tmdb",
        "movie": recommendation_service.get_movie_row(movie.tmdb_id),
    }


@router.delete("/{movie_id}/library")
def remove_tmdb_movie_from_library(movie_id: int, db: Session = Depends(get_db)):
    """Remove a movie explicitly imported from TMDB.

    Original CSV movies are immutable and cannot be removed through this
    endpoint. User watchlist/feedback rows are intentionally left intact so
    history is not silently destroyed.
    """
    from app.db.models import LibraryMovie

    movie = db.get(LibraryMovie, movie_id)
    if movie is None:
        if recommendation_service.is_in_base_catalogue(movie_id):
            raise HTTPException(status_code=409, detail="Original catalogue movies cannot be removed.")
        raise HTTPException(status_code=404, detail="Movie is not in the imported TMDB library.")

    db.delete(movie)
    db.commit()
    recommendation_service.refresh()
    return {"status": "removed", "movie_id": movie_id}


@router.get("/{movie_id}")
def get_movie(movie_id: int):
    """Return either an original catalogue movie or a persisted TMDB movie."""
    row = recommendation_service.get_movie_row(movie_id)
    if row is not None:
        return row
    raise HTTPException(status_code=404, detail="Movie is not in the current library.")


@router.get("/{movie_id}/tmdb-detail")
async def get_tmdb_detail(movie_id: int):
    """Full live TMDB detail, even when the movie has not been imported yet."""
    try:
        data = await tmdb_client.get_movie(movie_id)
        row = tmdb_movie_to_row(data)
        return {**row, "source": "tmdb_external"}
    except TMDBError as e:
        raise HTTPException(status_code=502, detail=str(e))
