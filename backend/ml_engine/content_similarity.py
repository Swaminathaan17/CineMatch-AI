"""High-precision content recommendation engine.

Phase 5-quality retrieval improvements applied to Phase 3/4:
- structured metadata and plot overview are represented separately so long plots
  cannot drown out strong entity matches such as director/genre/cast.
- word n-grams improve phrase-level matching while sublinear TF-IDF reduces the
  effect of repeated/common terms.
- structured fields are deliberately weighted and exact shared entities are
  exposed for explanations.
- external TMDB movies use the exact same dual-vector representation.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _clean_list_field(value) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str) and value.strip():
        items = [v.strip() for v in value.split(",")]
    else:
        items = []
    return [item for item in items if item]


def _soup_tokens(value) -> list[str]:
    return [item.replace(" ", "") for item in _clean_list_field(value)]


def build_soup(row: pd.Series) -> str:
    genres = _soup_tokens(row.get("genres"))
    cast = _soup_tokens(row.get("cast"))[:5]
    director = _soup_tokens(row.get("director"))
    keywords = _soup_tokens(row.get("keywords"))
    # Structured metadata gets much more weight than the long tail of keywords.
    return " ".join((genres * 4) + (director * 4) + (cast * 2) + keywords).lower()


def _overview(row: pd.Series) -> str:
    return str(row.get("overview") or "").strip().lower()


class ContentSimilarityEngine:
    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.reset_index(drop=True).copy()
        for col in ("genres", "cast", "director", "keywords", "overview"):
            if col not in self.movies_df.columns:
                self.movies_df[col] = ""
        self.movies_df["soup"] = self.movies_df.apply(build_soup, axis=1)
        self.movies_df["overview_text"] = self.movies_df.apply(_overview, axis=1)

        # Separate spaces prevent a 300-word plot from overwhelming genre/director.
        self.meta_vectorizer = TfidfVectorizer(
            stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=1
        )
        self.overview_vectorizer = TfidfVectorizer(
            stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=1
        )
        self.meta_matrix = self.meta_vectorizer.fit_transform(self.movies_df["soup"])
        if self.movies_df["overview_text"].str.strip().ne("").any():
            self.overview_matrix = self.overview_vectorizer.fit_transform(self.movies_df["overview_text"])
            self._has_overview_vocab = True
        else:
            self.overview_matrix = csr_matrix((len(self.movies_df), 1), dtype=float)
            self._has_overview_vocab = False
        # Kept for backward compatibility with personalization code/tests.
        self.tfidf_matrix = self.meta_matrix
        self.vectorizer = self.meta_vectorizer

        self._id_to_index = {movie_id: idx for idx, movie_id in enumerate(self.movies_df["id"])}

    @staticmethod
    def _combine(meta_scores, overview_scores):
        # Metadata is the primary signal; overview supplies semantic context.
        return 0.72 * meta_scores + 0.28 * overview_scores

    def _similarity_to_index(self, idx: int) -> np.ndarray:
        meta = cosine_similarity(self.meta_matrix[idx], self.meta_matrix)[0]
        overview = cosine_similarity(self.overview_matrix[idx], self.overview_matrix)[0] if self._has_overview_vocab else np.zeros(len(self.movies_df))
        return self._combine(meta, overview)

    def get_similar(self, movie_id: int, top_n: int = 10) -> list[dict]:
        if movie_id not in self._id_to_index:
            raise ValueError(f"movie_id {movie_id} not in dataset")
        idx = self._id_to_index[movie_id]
        scores = self._similarity_to_index(idx)
        ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
        top_matches = [s for s in ranked if s[0] != idx][:top_n]
        return [self._result(idx, match_idx, float(score), self.movies_df.iloc[idx])
                for match_idx, score in top_matches]

    def get_similar_to_external(self, movie: dict, top_n: int = 10) -> list[dict]:
        query = pd.Series(movie)
        meta_vec = self.meta_vectorizer.transform([build_soup(query)])
        meta = cosine_similarity(meta_vec, self.meta_matrix)[0]
        if self._has_overview_vocab and _overview(query):
            overview_vec = self.overview_vectorizer.transform([_overview(query)])
            overview = cosine_similarity(overview_vec, self.overview_matrix)[0]
        else:
            overview = np.zeros(len(self.movies_df))
        scores = self._combine(meta, overview)
        ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)[:top_n]
        return [self._result_external(query, match_idx, float(score)) for match_idx, score in ranked]

    def similarity_between_ids(self, movie_id_a: int, movie_id_b: int) -> float:
        """Return content similarity for diversity-aware re-ranking (MMR)."""
        if movie_id_a not in self._id_to_index or movie_id_b not in self._id_to_index:
            return 0.0
        a, b = self._id_to_index[movie_id_a], self._id_to_index[movie_id_b]
        meta = cosine_similarity(self.meta_matrix[a], self.meta_matrix[b])[0, 0]
        overview = cosine_similarity(self.overview_matrix[a], self.overview_matrix[b])[0, 0] if self._has_overview_vocab else 0.0
        return float(self._combine(meta, overview))

    def _result(self, query_idx, match_idx, score, query_row):
        row = self.movies_df.iloc[match_idx]
        return {
            "id": int(row["id"]), "title": row["title"],
            "similarity_score": round(score, 4),
            "shared_genres": self._shared_values(query_row, row, "genres"),
            "shared_cast": self._shared_values(query_row, row, "cast"),
            "shared_director": self._shared_values(query_row, row, "director"),
        }

    def _result_external(self, query_row, match_idx, score):
        row = self.movies_df.iloc[match_idx]
        return {
            "id": int(row["id"]), "title": row["title"],
            "similarity_score": round(score, 4),
            "shared_genres": self._shared_values(query_row, row, "genres"),
            "shared_cast": self._shared_values(query_row, row, "cast"),
            "shared_director": self._shared_values(query_row, row, "director"),
        }

    @staticmethod
    def _shared_values(row_a: pd.Series, row_b: pd.Series, field: str) -> list[str]:
        return sorted(set(_clean_list_field(row_a.get(field))) & set(_clean_list_field(row_b.get(field))))

    def _shared_field(self, idx_a: int, idx_b: int, field: str) -> list[str]:
        return self._shared_values(self.movies_df.iloc[idx_a], self.movies_df.iloc[idx_b], field)
