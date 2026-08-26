"""
Builds a user preference profile from the movies they've liked (weighted
average of those movies' TF-IDF vectors, reusing the exact same vector space
ContentSimilarityEngine already built - no separate embedding step needed)
plus an optional boost for explicitly favorited genres.

Cold start (no liked movies, no genre preferences) is handled by the caller
falling back to popularity - this module only knows how to do the vector math,
it doesn't decide what to show a brand new user.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ml_engine.content_similarity import ContentSimilarityEngine


class PersonalizationEngine:
    def __init__(self, engine: ContentSimilarityEngine):
        self.engine = engine

    def has_enough_signal(self, liked_movie_ids: list[int]) -> bool:
        valid = [mid for mid in liked_movie_ids if mid in self.engine._id_to_index]
        return len(valid) > 0

    def build_profile_vector(
        self, liked_movie_ids: list[int], favorite_genres: list[str] | None = None
    ) -> np.ndarray | None:
        valid_indices = [
            self.engine._id_to_index[mid]
            for mid in liked_movie_ids
            if mid in self.engine._id_to_index
        ]
        if not valid_indices:
            return None

        liked_vectors = self.engine.tfidf_matrix[valid_indices]
        profile = np.asarray(liked_vectors.mean(axis=0))

        if favorite_genres:
            feature_names = self.engine.vectorizer.get_feature_names_out()
            genre_tokens = {g.replace(" ", "").lower() for g in favorite_genres}
            for i, term in enumerate(feature_names):
                if term in genre_tokens:
                    # boost genre terms so explicit preferences count for
                    # more than whatever happened to show up in liked movies
                    profile[0, i] *= 1.5

        return profile

    def rank_for_user(
        self,
        liked_movie_ids: list[int],
        favorite_genres: list[str] | None = None,
        top_n: int = 10,
    ) -> list[dict]:
        profile = self.build_profile_vector(liked_movie_ids, favorite_genres)
        if profile is None:
            return []

        similarities = cosine_similarity(profile, self.engine.tfidf_matrix)[0]
        liked_set = set(liked_movie_ids)

        scored = []
        for idx, score in enumerate(similarities):
            row = self.engine.movies_df.iloc[idx]
            movie_id = int(row["id"])
            if movie_id in liked_set:
                continue  # don't recommend what they already liked
            scored.append((movie_id, row["title"], float(score)))

        scored.sort(key=lambda x: x[2], reverse=True)
        return [
            {"id": mid, "title": title, "preference_match_score": round(score, 4)}
            for mid, title, score in scored[:top_n]
        ]
