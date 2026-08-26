"""
Content-based similarity engine.

Takes a DataFrame of movie metadata, builds a combined text "soup" per movie
(genres + director + cast + keywords, overview handled separately for the
semantic/NL search in a later module), vectorizes it, and computes cosine
similarity so we can answer "movies similar to X".

Deliberately has zero FastAPI/TMDB imports - it should be usable and testable
as a plain Python module.
"""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _clean_list_field(value) -> list[str]:
    """Normalize a field that may be a list, comma string, or missing.
    Keeps readable spacing (e.g. "Christopher Nolan") - this is the version
    used for display / explainability."""
    if isinstance(value, list):
        items = value
    elif isinstance(value, str) and value.strip():
        items = [v.strip() for v in value.split(",")]
    else:
        items = []
    return [item for item in items if item]


def _soup_tokens(value) -> list[str]:
    """Same as _clean_list_field but with spaces stripped inside each token
    (e.g. "ChristopherNolan") so the vectorizer doesn't treat "Christopher"
    as a signal shared with every other "Christopher" in the dataset. Only
    used for building the TF-IDF soup, never shown to a user."""
    return [item.replace(" ", "") for item in _clean_list_field(value)]


def build_soup(row: pd.Series) -> str:
    """Combine a movie's structured metadata into one weighted text blob."""
    genres = _soup_tokens(row.get("genres"))
    cast = _soup_tokens(row.get("cast"))[:5]  # top 5 billed only
    director = _soup_tokens(row.get("director"))
    keywords = _soup_tokens(row.get("keywords"))

    # director/genre repeated to weight them higher than the long tail of
    # keywords - a naive single-count soup lets keyword volume drown out
    # the signals that actually matter most for "similar movie" judgments
    parts = (
        genres * 3
        + director * 3
        + cast * 2
        + keywords
    )
    return " ".join(parts).lower()


class ContentSimilarityEngine:
    def __init__(self, movies_df: pd.DataFrame):
        """movies_df must have: id, title, genres, cast, director, keywords"""
        self.movies_df = movies_df.reset_index(drop=True)
        self.movies_df["soup"] = self.movies_df.apply(build_soup, axis=1)

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df["soup"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        self._id_to_index = {
            movie_id: idx for idx, movie_id in enumerate(self.movies_df["id"])
        }

    def get_similar(self, movie_id: int, top_n: int = 10) -> list[dict]:
        if movie_id not in self._id_to_index:
            raise ValueError(f"movie_id {movie_id} not in dataset")

        idx = self._id_to_index[movie_id]
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        # skip index 0 - that's the movie itself (similarity == 1.0)
        top_matches = [s for s in scores if s[0] != idx][:top_n]

        results = []
        for match_idx, score in top_matches:
            row = self.movies_df.iloc[match_idx]
            results.append(
                {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "similarity_score": round(float(score), 4),
                    "shared_genres": self._shared_field(idx, match_idx, "genres"),
                    "shared_cast": self._shared_field(idx, match_idx, "cast"),
                    "shared_director": self._shared_field(idx, match_idx, "director"),
                }
            )
        return results

    def get_similar_to_external(self, movie: dict, top_n: int = 10) -> list[dict]:
        """Compare an external movie (for example a TMDB movie) against the
        local catalogue without adding it to the training dataset.

        This is the key Phase 1 bridge: TMDB supplies the movie metadata,
        while our existing TF-IDF vocabulary remains the recommendation
        engine. The external movie is only a temporary query vector.
        """
        query_row = pd.Series(movie)
        query_soup = build_soup(query_row)
        query_vector = self.vectorizer.transform([query_soup])
        scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        top_indices = sorted(
            enumerate(scores), key=lambda x: float(x[1]), reverse=True
        )[:top_n]

        results = []
        for match_idx, score in top_indices:
            row = self.movies_df.iloc[match_idx]
            results.append(
                {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "similarity_score": round(float(score), 4),
                    "shared_genres": self._shared_values(query_row, row, "genres"),
                    "shared_cast": self._shared_values(query_row, row, "cast"),
                    "shared_director": self._shared_values(query_row, row, "director"),
                }
            )
        return results

    @staticmethod
    def _shared_values(row_a: pd.Series, row_b: pd.Series, field: str) -> list[str]:
        set_a = set(_clean_list_field(row_a.get(field)))
        set_b = set(_clean_list_field(row_b.get(field)))
        return sorted(set_a & set_b)

    def _shared_field(self, idx_a: int, idx_b: int, field: str) -> list[str]:
        """Used by the explainability layer later - what specifically overlapped."""
        set_a = set(_clean_list_field(self.movies_df.iloc[idx_a].get(field)))
        set_b = set(_clean_list_field(self.movies_df.iloc[idx_b].get(field)))
        return sorted(set_a & set_b)
