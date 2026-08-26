"""Recommendation service backed by the original catalogue plus Phase 2 library movies."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app.db.models import LibraryMovie
from app.db.session import SessionLocal
from app.services.library_service import library_movie_to_row
from ml_engine.content_similarity import ContentSimilarityEngine
from ml_engine.personalization import PersonalizationEngine
from ml_engine.nl_query_parser import NLQueryEngine

DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data"


class RecommendationService:
    def __init__(self):
        self._engine: ContentSimilarityEngine | None = None
        self._personalization: PersonalizationEngine | None = None
        self._nl_engine: NLQueryEngine | None = None
        self._df: pd.DataFrame | None = None

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["genres", "cast", "director", "keywords"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.replace(r"[\[\]'\"]", "", regex=True)
        if "source" not in df.columns:
            df["source"] = "local"
        return df

    def _load(self):
        if self._engine is not None:
            return

        seeded = DATA_PATH / "movies_dataset.csv"
        sample = DATA_PATH / "sample_movies.csv"
        path = seeded if seeded.exists() else sample

        base_df = pd.read_csv(path)
        base_df = self._normalize_df(base_df)
        base_df["source"] = "local"

        # Phase 2: restore every movie the user explicitly imported from TMDB.
        # This makes the growing catalogue survive backend restarts.
        db = SessionLocal()
        try:
            dynamic_rows = [library_movie_to_row(m) for m in db.query(LibraryMovie).all()]
        finally:
            db.close()

        if dynamic_rows:
            dynamic_df = self._normalize_df(pd.DataFrame(dynamic_rows))
            dynamic_df["source"] = "tmdb"
            # The base dataset is authoritative. If a TMDB movie is already in
            # the original catalogue, never create a duplicate vector/id.
            dynamic_df = dynamic_df[~dynamic_df["id"].isin(set(base_df["id"]))]
            if not dynamic_df.empty:
                # Keep a stable union of columns; missing metadata is harmless.
                base_df = pd.concat([base_df, dynamic_df], ignore_index=True, sort=False)

        self._df = base_df
        self._engine = ContentSimilarityEngine(self._df)
        self._personalization = PersonalizationEngine(self._engine)
        self._nl_engine = NLQueryEngine(self._df)

    def refresh(self):
        """Rebuild the in-memory TF-IDF/NL indexes after a library mutation."""
        self._engine = None
        self._personalization = None
        self._nl_engine = None
        self._df = None
        self._load()

    def is_in_base_catalogue(self, movie_id: int) -> bool:
        self._load()
        # The source field lets us distinguish a CSV movie from a persisted
        # TMDB import even though both use TMDB's numeric movie id.
        match = self._df[self._df["id"] == movie_id]
        return not match.empty and match.iloc[0].get("source") == "local"

    def search_by_mood(self, query: str, top_n: int = 10) -> dict:
        self._load()
        return self._nl_engine.search(query, top_n=top_n)

    def get_similar(self, movie_id: int, top_n: int = 10) -> list[dict]:
        self._load()
        return self._engine.get_similar(movie_id, top_n=top_n)

    def get_similar_to_external(self, movie: dict, top_n: int = 10) -> list[dict]:
        """Recommend library movies for an external TMDB movie without saving it."""
        self._load()
        return self._engine.get_similar_to_external(movie, top_n=top_n)

    def get_personalized(self, liked_movie_ids: list[int], favorite_genres: list[str], top_n: int = 10) -> list[dict]:
        self._load()
        return self._personalization.rank_for_user(liked_movie_ids, favorite_genres, top_n=top_n)

    def get_trending_fallback(self, top_n: int = 10) -> list[dict]:
        self._load()
        df = self._df.copy()
        if "vote_average" not in df.columns:
            df["vote_average"] = 0
        if "popularity" not in df.columns:
            df["popularity"] = 0
        df["_trending_score"] = df["vote_average"].fillna(0) * 0.6 + df["popularity"].fillna(0) * 0.4
        top = df.sort_values("_trending_score", ascending=False).head(top_n)
        return [
            {
                "id": int(r["id"]),
                "title": r["title"],
                "poster_path": r.get("poster_path"),
                "source": r.get("source", "local"),
            }
            for _, r in top.iterrows()
        ]

    def get_movie_row(self, movie_id: int) -> dict | None:
        self._load()
        match = self._df[self._df["id"] == movie_id]
        if match.empty:
            return None
        row = match.iloc[0].to_dict()
        row["id"] = int(row["id"])
        row["source"] = row.get("source", "local")
        return row

    def list_movies(self) -> list[dict]:
        self._load()
        cols = [c for c in ["id", "title", "poster_path", "backdrop_path", "release_date", "source"] if c in self._df.columns]
        return self._df[cols].to_dict(orient="records")

    def search_local(self, query: str, top_n: int = 10) -> list[dict]:
        self._load()
        import difflib

        query_lower = query.lower().strip()
        substring_matches = self._df[
            self._df["title"].astype(str).str.lower().str.contains(query_lower, regex=False, na=False)
        ]

        if len(substring_matches) > 0:
            rows = substring_matches.head(top_n)
        else:
            id_by_title = dict(zip(self._df["title"], self._df["id"]))
            lowered_to_original = {t.lower(): t for t in id_by_title}
            close_lowered = difflib.get_close_matches(
                query_lower, list(lowered_to_original.keys()), n=top_n, cutoff=0.75
            )
            close = [lowered_to_original[t] for t in close_lowered]
            rows = self._df[self._df["title"].isin(close)].head(top_n)

        return [
            {
                "id": int(r["id"]),
                "title": r["title"],
                "source": r.get("source", "local"),
                "poster_path": r.get("poster_path"),
                "release_date": r.get("release_date", ""),
            }
            for _, r in rows.iterrows()
        ]


recommendation_service = RecommendationService()
