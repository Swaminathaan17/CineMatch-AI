"""
Run this once (in Replit, where TMDB is reachable) to build a starter dataset:

    python ml_training/seed_dataset.py --pages 20

Pulls ~20 pages x 20 movies = 400 popular movies, fetches full detail
(credits + keywords) for each, and saves to data/movies_dataset.csv.

This is intentionally a script, not part of the served app - dataset building
is a one-time/occasional job, not something that should run on every request.
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.services.tmdb_client import tmdb_client
from app.services.data_prep import build_dataset_from_tmdb


async def fetch_all(pages: int) -> list[dict]:
    movie_ids = []
    for page in range(1, pages + 1):
        result = await tmdb_client.get_popular(page=page)
        movie_ids.extend([m["id"] for m in result.get("results", [])])
        time.sleep(0.1)  # be polite to the API

    print(f"Found {len(movie_ids)} movie ids, fetching full detail for each...")

    full_movies = []
    for i, movie_id in enumerate(movie_ids):
        try:
            detail = await tmdb_client.get_movie(movie_id)
            full_movies.append(detail)
        except Exception as e:
            print(f"  skipped {movie_id}: {e}")
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(movie_ids)} done")
        time.sleep(0.05)

    return full_movies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=10)
    parser.add_argument("--out", default="data/movies_dataset.csv")
    args = parser.parse_args()

    movies = asyncio.run(fetch_all(args.pages))
    df = build_dataset_from_tmdb(movies)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} movies to {args.out}")


if __name__ == "__main__":
    main()
