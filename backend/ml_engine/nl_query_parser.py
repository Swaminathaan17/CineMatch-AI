"""
Natural language / mood search: "I want a dark psychological thriller with a
surprising ending" -> ranked movies.

Two-tier design:
  1. PREFERRED: sentence-transformers (all-MiniLM-L6-v2) encodes the query and
     each movie's overview into the same semantic vector space, so "mind-bending"
     can match "reality-bending" even with zero shared keywords. This needs to
     download its model weights from Hugging Face the first time it runs.
  2. FALLBACK: if sentence-transformers isn't installed or the model can't be
     downloaded (e.g. no internet), degrade to TF-IDF cosine similarity over
     the same overview text. This is real keyword-level matching, not fake -
     it just can't catch pure semantic paraphrase the way embeddings can.

Whichever mode is active is reported in every result so the frontend/you can
be honest about which one produced a given match, instead of silently
pretending both modes are equivalent.
"""
from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MOOD_GENRE_HINTS = {
    "funny": ["Comedy"], "hilarious": ["Comedy"], "light": ["Comedy", "Romance"],
    "dark": ["Thriller", "Crime", "Horror"], "scary": ["Horror"], "creepy": ["Horror"],
    "romantic": ["Romance"], "heartwarming": ["Drama", "Romance"],
    "mind-bending": ["Science Fiction", "Mystery"], "twist": ["Mystery", "Thriller"],
    "emotional": ["Drama"], "sad": ["Drama"], "uplifting": ["Drama", "Comedy"],
    "action-packed": ["Action"], "intense": ["Thriller", "Action"],
    "psychological": ["Thriller", "Mystery"], "epic": ["Adventure", "Fantasy"],

    # content-word -> genre mappings (as opposed to mood-adjectives above) -
    # added after finding queries like "astronauts stranded on mars" got no
    # hint at all (no mood adjective present) and fell back to pure noisy
    # TF-IDF, same failure mode this whole mechanism exists to fix
    "space": ["Science Fiction"], "astronaut": ["Science Fiction"], "alien": ["Science Fiction"],
    "robot": ["Science Fiction"], "future": ["Science Fiction"], "mars": ["Science Fiction"],
    "heist": ["Crime", "Thriller"], "detective": ["Crime", "Mystery"], "murder": ["Crime", "Mystery"],
    "war": ["War"], "soldier": ["War"], "battle": ["War", "Action"],
    "zombie": ["Horror"], "ghost": ["Horror"], "haunted": ["Horror"], "monster": ["Horror"],
    "wizard": ["Fantasy"], "magic": ["Fantasy"], "dragon": ["Fantasy"],
    "superhero": ["Action", "Fantasy"], "musical": ["Music"], "dance": ["Music"],
    "cowboy": ["Western"], "sports": ["Sport"], "wedding": ["Romance", "Comedy"],
    "survival": ["Adventure", "Thriller"], "true story": ["Drama", "History"],
}


def _detect_literal_genre_mentions(query: str) -> list[str]:
    """Catches queries that name a genre directly ('a sci-fi movie', 'a
    good horror flick') even when no mood adjective from the dict above is
    present - a second, simpler detection layer rather than relying on one
    keyword dictionary to catch everything. Checks both the dataset's real
    genre string ("Science Fiction") and the casual way people actually
    type it ("sci-fi")."""
    known_genres = [
        "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
        "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
        "Romance", "Science Fiction", "Thriller", "War", "Western",
    ]
    lowered = query.lower()
    found = [g for g in known_genres if g.lower() in lowered]
    if "sci-fi" in lowered or "scifi" in lowered:
        found.append("Science Fiction")
    return list(set(found))


def extract_mood_hints(query: str) -> list[str]:
    """Genre hints used both for the human-readable 'AI interpretation'
    summary and (in fallback mode) to boost ranking - combines mood-adjective
    matches ("dark", "twist") with literal genre mentions ("a sci-fi movie")."""
    lowered = query.lower()
    hints = set()
    for keyword, genres in MOOD_GENRE_HINTS.items():
        if keyword in lowered:
            hints.update(genres)
    hints.update(_detect_literal_genre_mentions(query))
    return sorted(hints)


class NLQueryEngine:
    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.reset_index(drop=True)
        self.mode = "fallback"
        self._embedder = None
        self._movie_embeddings = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None

        self._try_load_semantic_model()
        self._build_fallback_index()

    def _try_load_semantic_model(self):
        try:
            from sentence_transformers import SentenceTransformer  # noqa: heavy import, only on this path

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            overviews = self.movies_df.get("overview", pd.Series([""] * len(self.movies_df))).fillna("")
            self._movie_embeddings = self._embedder.encode(overviews.tolist())
            self.mode = "semantic"
        except Exception:
            # covers: package not installed, no internet to fetch weights,
            # or any other load failure - fall back rather than crash
            self._embedder = None
            self.mode = "fallback"

    def _build_fallback_index(self):
        overviews = self.movies_df.get("overview", pd.Series([""] * len(self.movies_df))).fillna("")
        # if overview is missing/empty for every row (e.g. the 10-movie sample
        # dataset has no overview column), fall back to the genre/keyword soup
        text_source = overviews if overviews.str.strip().any() else self.movies_df.apply(
            lambda r: " ".join(
                str(r.get(c, "")) for c in ["genres", "keywords"] if c in r
            ),
            axis=1,
        )
        self._tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(text_source)

    def search(self, query: str, top_n: int = 10) -> dict:
        mood_hints = extract_mood_hints(query)

        if self.mode == "semantic" and self._embedder is not None:
            query_vec = self._embedder.encode([query])
            scores = cosine_similarity(query_vec, self._movie_embeddings)[0]
        else:
            query_vec = self._tfidf_vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self._tfidf_matrix)[0]

        # In fallback mode, raw TF-IDF over long free-text overviews is
        # noisy on its own - confirmed by real failures during testing
        # (a "heist movie with a twist" query surfaced "Disaster Movie",
        # a "dreams and reality" query surfaced "Boxing Helena", neither
        # sharing any real thematic connection to the query). The mood
        # hints were already being extracted for the "interpreted as"
        # display but never actually used to influence ranking - fixing
        # that gap here: boost movies whose genre overlaps the interpreted
        # mood, and in fallback mode only, apply a mild penalty to those
        # that share none of the interpreted genres at all. Semantic mode
        # doesn't need this crutch - embeddings already capture genre-like
        # signal implicitly.
        if self.mode == "fallback" and mood_hints:
            scores = self._apply_genre_boost(scores, mood_hints)

        ranked_idx = scores.argsort()[::-1][:top_n]
        results = [
            {
                "id": int(self.movies_df.iloc[i]["id"]),
                "title": self.movies_df.iloc[i]["title"],
                "match_score": round(float(scores[i]), 4),
            }
            for i in ranked_idx
        ]

        return {
            "mode": self.mode,
            "interpreted_genres": mood_hints,
            "results": results,
        }

    def _apply_genre_boost(self, scores, mood_hints: list[str]):
        mood_hint_set = {g.lower() for g in mood_hints}
        boosted = scores.copy()
        for i, genres_str in enumerate(self.movies_df.get("genres", pd.Series([""] * len(self.movies_df)))):
            movie_genres = {g.strip().lower() for g in str(genres_str).split(",")}
            if movie_genres & mood_hint_set:
                boosted[i] *= 1.6
            else:
                boosted[i] *= 0.5
        return boosted
