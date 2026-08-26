"""
Converts TMDB's raw JSON shape (movie detail + credits + keywords, fetched via
append_to_response) into the flat row format ml_engine.content_similarity expects.

Kept separate from tmdb_client.py on purpose: tmdb_client only knows how to talk
to TMDB's API, this module only knows how to reshape that data. Two reasons to
change, two files.
"""
from __future__ import annotations


def tmdb_movie_to_row(movie_json: dict) -> dict:
    genres = [g["name"] for g in movie_json.get("genres", [])]

    credits = movie_json.get("credits", {})
    cast = [c["name"] for c in credits.get("cast", [])[:10]]
    director = [
        c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"
    ]

    keywords_block = movie_json.get("keywords", {})
    # TMDB nests keywords differently depending on endpoint - "keywords" key
    # for movies, "results" for the standalone keywords endpoint
    raw_keywords = keywords_block.get("keywords") or keywords_block.get("results") or []
    keywords = [k["name"] for k in raw_keywords]

    return {
        "id": movie_json["id"],
        "title": movie_json.get("title", ""),
        "overview": movie_json.get("overview", ""),
        "genres": genres,
        "cast": cast,
        "director": director,
        "keywords": keywords,
        "poster_path": movie_json.get("poster_path"),
        "backdrop_path": movie_json.get("backdrop_path"),
        "release_date": movie_json.get("release_date", ""),
        "vote_average": movie_json.get("vote_average", 0),
        "popularity": movie_json.get("popularity", 0),
    }


def build_dataset_from_tmdb(movie_jsons: list[dict]):
    """Takes a list of raw TMDB movie-detail JSONs, returns a pandas DataFrame
    ready to hand to ContentSimilarityEngine."""
    import pandas as pd

    rows = [tmdb_movie_to_row(m) for m in movie_jsons]
    return pd.DataFrame(rows)
