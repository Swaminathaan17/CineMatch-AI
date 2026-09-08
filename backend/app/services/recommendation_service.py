"""Recommendation service backed by the original catalogue plus Phase 2 library movies."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from app.db.models import LibraryMovie, MovieAsset
from app.db.session import SessionLocal
from app.services.library_service import library_movie_to_row
from ml_engine.content_similarity import ContentSimilarityEngine
from ml_engine.personalization import PersonalizationEngine
from ml_engine.nl_query_parser import NLQueryEngine
from ml_engine.hybrid_ranker import compute_hybrid_score, recency_score
from ml_engine.recommendation_explainer import build_recommendation_explanations

DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data"


def _is_blank(value) -> bool:
    """True for None, NaN, and empty strings - used to decide whether a local
    asset field can be enriched without overwriting real data."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN is never equal to itself
        return True
    return str(value).strip() == ""


def _json_safe(value):
    """Convert a pandas/numpy cell into a plain JSON-friendly value.

    Merged rows (CSV ∪ library/asset overlays) can carry NaN/Inf floats in
    columns the other frame lacks; Starlette's JSON encoder (allow_nan=False)
    rejects those as 500s. Map non-finite floats to None and unwrap numpy
    scalars to native Python types.
    """
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
    return value


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
        # runtime only exists for TMDB-imported rows; base-CSV rows get NaN when
        # the frames are concatenated. Normalize to an int or None so the JSON
        # response never serializes a NaN (Starlette rejects those) and never
        # fabricates a runtime for catalogue movies.
        if "runtime" in df.columns:
            df["runtime"] = df["runtime"].apply(
                lambda v: int(v) if pd.notna(v) else None
            )
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
        self._apply_asset_overlay()
        self._engine = ContentSimilarityEngine(self._df)
        self._personalization = PersonalizationEngine(self._engine)
        self._nl_engine = NLQueryEngine(self._df)

    def _apply_asset_overlay(self):
        """Phase 1 Step 3: fill asset gaps for original-catalogue rows from the
        backfill enrichment table (movie_assets).

        Only blank values are overwritten - everything else on the local row
        stays authoritative. The base CSV has no poster/backdrop/release/runtime
        columns at all, so the columns are materialized here (None by default);
        before this, catalogue rows simply had no asset fields. If the table is
        missing (old DB not yet migrated), the overlay is a no-op.
        """
        try:
            db = SessionLocal()
            try:
                assets = db.query(MovieAsset).all()
            finally:
                db.close()
        except Exception:
            assets = []

        for col in ("poster_path", "backdrop_path", "release_date", "runtime"):
            if col not in self._df.columns:
                self._df[col] = None

        for asset in assets:
            match = self._df[self._df["id"] == asset.tmdb_id]
            if match.empty:
                continue
            idx = match.index[0]
            if _is_blank(self._df.at[idx, "poster_path"]):
                self._df.at[idx, "poster_path"] = asset.poster_path
            if _is_blank(self._df.at[idx, "backdrop_path"]):
                self._df.at[idx, "backdrop_path"] = asset.backdrop_path
            if _is_blank(self._df.at[idx, "release_date"]):
                self._df.at[idx, "release_date"] = asset.release_date
            if _is_blank(self._df.at[idx, "runtime"]):
                self._df.at[idx, "runtime"] = asset.runtime

        # Normalize NaN (created when frames with and without asset columns are
        # concatenated) to None so JSON serialization never hits a float NaN.
        for col in ("poster_path", "backdrop_path", "release_date", "runtime"):
            if col in self._df.columns:
                self._df[col] = self._df[col].apply(
                    lambda v: None if _is_blank(v) else v
                )

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

    def diversify(self, ranked: list[dict], top_n: int = 10, lambda_: float = 0.84) -> list[dict]:
        """MMR re-ranking: preserve relevance while removing near-duplicate results."""
        self._load()
        if len(ranked) <= top_n:
            return ranked[:top_n]
        remaining = ranked.copy()
        selected: list[dict] = []
        while remaining and len(selected) < top_n:
            best_idx, best_value = 0, float("-inf")
            for i, candidate in enumerate(remaining):
                relevance = float(candidate.get("_raw_score", candidate.get("match_percentage", 0) / 100))
                redundancy = 0.0
                if selected:
                    redundancy = max(
                        self._engine.similarity_between_ids(candidate["id"], chosen["id"])
                        for chosen in selected
                    )
                mmr = lambda_ * relevance - (1 - lambda_) * redundancy
                if mmr > best_value:
                    best_idx, best_value = i, mmr
            chosen = remaining.pop(best_idx)
            chosen["diversity_score"] = round(best_value, 4)
            chosen.pop("_raw_score", None)
            selected.append(chosen)
        return selected

    def get_personalized(self, liked_movie_ids: list[int], favorite_genres: list[str], top_n: int = 10) -> list[dict]:
        self._load()
        return self._personalization.rank_for_user(liked_movie_ids, favorite_genres, top_n=top_n)

    def rank_personalized_candidates(
        self,
        signals: dict,
        top_n: int = 10,
    ) -> list[dict]:
        """Phase 3 ranking for the home feed, driven by Phase 2.2 signals.

        Consumes a batched signal bundle (user_service.get_signal_bundle): a
        behavioral taste profile supplies per-movie preference_match, and the
        same bundle supplies direct per-movie nudges (ratings, watchlist, views,
        recommendation feedback). All signal loading happens once, up front - no
        database queries happen inside the ranking loop.

        Does not call external review APIs, keeping the home page fast; detailed
        movie pages can still use the full sentiment-aware hybrid endpoint.
        """
        self._load()
        downvoted = set(signals.get("downvoted") or [])
        feedback_map = {movie_id: fb for movie_id, fb, _ in signals.get("feedback", [])}
        preference_scores = self._personalization.behavior_scores(signals)
        behavior_boosts = self._personalization.movie_behavior_boosts(signals)
        weights = self._personalization.aggregate_signal_weights(signals)
        if not preference_scores:
            return []

        liked_set = set(signals.get("liked_ids") or [])
        df = self._df.copy()
        popularity_max = float(pd.to_numeric(df.get("popularity", 0), errors="coerce").fillna(0).max() or 1)
        ranked = []
        for _, row in df.iterrows():
            movie_id = int(row["id"])
            if movie_id in liked_set or movie_id in downvoted:
                continue
            pref = preference_scores.get(movie_id, 0.0)
            score = compute_hybrid_score(
                content_similarity=0.0,
                sentiment_positive_pct=None,
                rating=float(row.get("vote_average") or 0),
                popularity=float(row.get("popularity") or 0),
                popularity_max_in_batch=popularity_max,
                preference_match=pref,
                recency=recency_score(row.get("release_date")),
                feedback_boost=behavior_boosts.get(movie_id, 0.0),
            )
            ranked.append({
                "id": movie_id,
                "title": row["title"],
                "poster_path": row.get("poster_path"),
                "source": row.get("source", "local"),
                "preference_match_score": round(pref, 4),
                "match_percentage": min(99, round(score["final_score"] * 100)),
                "score_breakdown": score["components"],
                "reasons": [
                    "Strong match with your taste profile" if pref >= 0.65 else "Matches patterns from movies you like",
                    "Highly rated by audiences" if float(row.get("vote_average") or 0) >= 7.5 else "Balanced with audience rating",
                ],
            })
        ranked.sort(key=lambda r: r["match_percentage"], reverse=True)
        for item in ranked:
            item["_raw_score"] = item["match_percentage"] / 100
        selected = self.diversify(ranked, top_n=top_n)
        # Transparent "why this movie?" per result, from the same signal bundle
        # and weights the ranking already used - no extra DB/API work.
        explained = build_recommendation_explanations(
            selected,
            self._personalization,
            signals,
            weights=weights,
            preference_scores=preference_scores,
            behavior_boosts=behavior_boosts,
            mode="personalized",
        )
        for item, explanation in zip(selected, explained):
            item["explanation"] = explanation
        return selected

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
                "explanation": {"primary": "Popular right now", "secondary": None},
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
        return {k: _json_safe(v) for k, v in row.items()}

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
