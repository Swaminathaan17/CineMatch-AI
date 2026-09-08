"""
Builds a user preference profile from behavioral signals - likes, ratings,
watchlist, views, searches and recommendation feedback - as a weighted average
of those movies' TF-IDF vectors, reusing the exact same vector space
ContentSimilarityEngine already built - no separate embedding step needed.
Plus an optional boost for explicitly favorited genres.

Two entry points:

- build_profile_vector / preference_scores / rank_for_user: the original
  like-only pipeline, kept for backward compatibility.
- build_behavior_profile / behavior_scores / movie_behavior_boosts: Phase 2.2
  path that consumes the full signal bundle (see user_service.get_signal_bundle)
  with recency decay and per-movie boosts. The old methods are deliberately
  unchanged so existing callers and tests keep working.

Every signal is weighted, recency-decayed (half-life in days), and per-movie
totals are magnitude-capped so stacking several positive signals about the same
movie never produces unreasonable double-counting - while a strong negative
(rating 1 star, downvoted) still pulls the profile away.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ml_engine.content_similarity import ContentSimilarityEngine
from ml_engine.nl_query_parser import extract_mood_hints

# ---------------------------------------------------------------------------
# Signal weights (transparent, tune here)
# ---------------------------------------------------------------------------

LIKE_WEIGHT = 1.0
WATCHLIST_WEIGHT = 0.5
VIEW_WEIGHT = 0.15
VIEW_COUNT_CAP = 3                 # at most 3 recent views count full-strength
FEEDBACK_DOWN_WEIGHT = -1.0
FEEDBACK_UP_WEIGHT = 0.3
RATING_TO_WEIGHT = {1: -1.0, 2: -0.5, 3: 0.1, 4: 0.5, 5: 1.0}

# Magnitude cap applied per movie after combining all signals. A movie that is
# liked, 5-star rated, on the watchlist and viewed repeatedly cannot exceed
# PROFILE_WEIGHT_CAP total, so positive signals keep a bounded footprint while
# negatives can still reach -PROFILE_WEIGHT_CAP.
PROFILE_WEIGHT_CAP = 1.5
MIN_SIGNAL_WEIGHT = 0.05

# Recency decay: 0.5 ** (age_days / half_life). Recent = full, older shrinks,
# and SIGNAL_DECAY_FLOOR keeps long-ago signals at a small-but-nonzero weight.
SIGNAL_HALF_LIFE_DAYS = 45
SIGNAL_DECAY_FLOOR = 0.1

UPDATE_SEARCH_WINDOW_DAYS = 90
SEARCH_GENRE_BOOST = 1.2
FAVORITE_GENRE_BOOST = 1.5

# Per-movie final-score nudges, all clamped into the same +/-0.08 band the
# existing hybrid ranker already reserves for feedback_boost.
RATING_NUDGE = 0.06
WATCHLIST_NUDGE = 0.04
VIEW_NUDGE = 0.02
VIEW_NUDGE_CAP = 0.04
FEEDBACK_UP_NUDGE = 0.03
FEEDBACK_DOWN_NUDGE = -0.08
BEHAVIOR_BOOST_CAP = 0.08


def _as_utc(value):
    """SQLite returns naive datetimes; normalize to UTC-aware for decay math."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def signal_decay(created_at, now=None, half_life_days=SIGNAL_HALF_LIFE_DAYS, floor=SIGNAL_DECAY_FLOOR) -> float:
    """Exponential recency decay: 1.0 at time zero, ~0.5 at the half-life,
    and never below `floor` so very old behavior keeps a small influence."""
    created_at = _as_utc(created_at)
    if created_at is None:
        return 1.0
    now = now or datetime.now(timezone.utc)
    age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    return max(floor, 0.5 ** (age_days / half_life_days))


def _genre_tokens(genres: list[str]) -> set[str]:
    return {g.replace(" ", "").lower() for g in genres if g}


class PersonalizationEngine:
    def __init__(self, engine: ContentSimilarityEngine):
        self.engine = engine

    def has_enough_signal(self, liked_movie_ids: list[int]) -> bool:
        valid = [mid for mid in liked_movie_ids if mid in self.engine._id_to_index]
        return len(valid) > 0

    def build_profile_vector(
        self,
        liked_movie_ids: list[int],
        favorite_genres: list[str] | None = None,
        negative_movie_ids: list[int] | None = None,
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

        # Explicit negative feedback should push the profile away from those
        # movies.  A modest subtraction is intentional: one accidental dislike
        # must not erase a strong positive taste signal.
        negative_indices = [
            self.engine._id_to_index[mid]
            for mid in (negative_movie_ids or [])
            if mid in self.engine._id_to_index
        ]
        if negative_indices:
            negative_profile = np.asarray(self.engine.tfidf_matrix[negative_indices].mean(axis=0))
            profile = np.maximum(profile - (0.45 * negative_profile), 0.0)

        if favorite_genres:
            feature_names = self.engine.vectorizer.get_feature_names_out()
            genre_tokens = {g.replace(" ", "").lower() for g in favorite_genres}
            for i, term in enumerate(feature_names):
                if term in genre_tokens:
                    # boost genre terms so explicit preferences count for
                    # more than whatever happened to show up in liked movies
                    profile[0, i] *= 1.5

        return profile

    def preference_scores(
        self,
        liked_movie_ids: list[int],
        favorite_genres: list[str] | None = None,
        negative_movie_ids: list[int] | None = None,
    ) -> dict[int, float]:
        """Return a normalized 0-1 taste similarity for every movie."""
        profile = self.build_profile_vector(liked_movie_ids, favorite_genres, negative_movie_ids)
        if profile is None:
            return {}
        similarities = cosine_similarity(profile, self.engine.tfidf_matrix)[0]
        return {int(self.engine.movies_df.iloc[i]["id"]): float(max(0.0, min(1.0, similarities[i])))
                for i in range(len(similarities))}

    def rank_for_user(
        self,
        liked_movie_ids: list[int],
        favorite_genres: list[str] | None = None,
        top_n: int = 10,
        negative_movie_ids: list[int] | None = None,
    ) -> list[dict]:
        profile = self.build_profile_vector(liked_movie_ids, favorite_genres, negative_movie_ids)
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

    # ------------------------------------------------------------------
    # Phase 2.2 - behavioral signal weighting (ratings, views, watchlist,
    # searches, feedback) on top of the same vector space.
    # ------------------------------------------------------------------

    def aggregate_signal_weights(self, bundle: dict) -> dict[int, float]:
        """Combine every signal in a user bundle into one weight per movie.

        Each contribution is recency-decayed; per-movie totals are clamped to
        +/- PROFILE_WEIGHT_CAP so repeated positives about the same movie do
        not produce unreasonable double-counting.
        """
        now = datetime.now(timezone.utc)
        weights: dict[int, float] = {}

        def add(movie_id: int, w: float) -> None:
            weights[movie_id] = weights.get(movie_id, 0.0) + w

        for movie_id, ts in bundle.get("liked", []):
            add(movie_id, LIKE_WEIGHT * signal_decay(ts, now))
        for movie_id, rating, ts in bundle.get("ratings", []):
            rw = RATING_TO_WEIGHT.get(int(round(rating)), 0.0)
            if rw:
                add(movie_id, rw * signal_decay(ts, now))
        for movie_id, ts in bundle.get("watchlist", []):
            add(movie_id, WATCHLIST_WEIGHT * signal_decay(ts, now))
        for movie_id, fb, ts in bundle.get("feedback", []):
            if fb == "down":
                add(movie_id, FEEDBACK_DOWN_WEIGHT * signal_decay(ts, now))
            elif fb == "up":
                add(movie_id, FEEDBACK_UP_WEIGHT * signal_decay(ts, now))

        # Views: repeated *recent* views stack up to VIEW_COUNT_CAP full-strength
        # contributions; beyond that they stop adding so random page reloads of
        # the same movie cannot rival an explicit like.
        view_totals: dict[int, float] = {}
        for movie_id, ts in bundle.get("views", []):
            view_totals[movie_id] = view_totals.get(movie_id, 0.0) + VIEW_WEIGHT * signal_decay(ts, now)
        for movie_id, total in view_totals.items():
            add(movie_id, min(total, VIEW_WEIGHT * VIEW_COUNT_CAP))

        return {
            movie_id: max(-PROFILE_WEIGHT_CAP, min(PROFILE_WEIGHT_CAP, w))
            for movie_id, w in weights.items()
        }

    def build_behavior_profile(self, bundle: dict) -> np.ndarray | None:
        """Weighted-average taste vector from the full signal bundle, L2
        normalized, with genre boosts from favorite genres and recent searches."""
        weights = self.aggregate_signal_weights(bundle)
        return self._build_profile(weights, bundle)

    def _build_profile(self, weights: dict, bundle: dict) -> np.ndarray | None:
        valid = {
            movie_id: w
            for movie_id, w in weights.items()
            if movie_id in self.engine._id_to_index and abs(w) >= MIN_SIGNAL_WEIGHT
        }
        if not valid:
            return None

        profile = sum(
            w * self.engine.tfidf_matrix[self.engine._id_to_index[movie_id]]
            for movie_id, w in valid.items()
        )
        profile = np.asarray(profile.toarray()).reshape(1, -1)

        feature_names = self.engine.vectorizer.get_feature_names_out()
        favorite_tokens = _genre_tokens(bundle.get("favorite_genres", []))
        if favorite_tokens:
            for i, term in enumerate(feature_names):
                if term in favorite_tokens:
                    profile[0, i] *= FAVORITE_GENRE_BOOST

        search_tokens = _genre_tokens(self.search_genre_hints(bundle))
        if search_tokens:
            for i, term in enumerate(feature_names):
                if term in search_tokens:
                    # weaker than explicit favorite genres - searches are only a
                    # conservative hint, never treated as a hard preference
                    profile[0, i] *= SEARCH_GENRE_BOOST

        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
        return profile

    def search_genre_hints(self, bundle: dict) -> list[str]:
        """Conservative genre hints extracted from recent searches.

        Only searches inside UPDATE_SEARCH_WINDOW_DAYS are considered and the
        existing mood/genre extractor (mood adjectives + literal genre
        mentions) is reused - raw free-text is never treated as a preference.
        """
        now = datetime.now(timezone.utc)
        window = []
        for query, ts in bundle.get("searches", []):
            ts = _as_utc(ts)
            if ts is None:
                continue
            age_days = (now - ts).total_seconds() / 86400.0
            if age_days <= UPDATE_SEARCH_WINDOW_DAYS:
                window.append(query)
        hints: set[str] = set()
        for query in window[:5]:
            hints.update(extract_mood_hints(query))
        return sorted(hints)

    def behavior_scores(self, bundle: dict) -> dict[int, float]:
        """Normalized 0-1 taste similarity for every movie from the signal
        bundle's profile. Empty dict when there is not enough signal.

        Cosine is scale-invariant, so a single like and a single view would
        otherwise produce identical results - multiply by the user's total
        behavioral strength so strong signals (a like, a 5-star) outscore
        weak ones (a view, an old event).
        """
        weights = self.aggregate_signal_weights(bundle)
        profile = self._build_profile(weights, bundle)
        if profile is None:
            return {}
        strength = min(1.0, sum(abs(w) for w in weights.values()))
        similarities = cosine_similarity(profile, self.engine.tfidf_matrix)[0]
        return {
            int(self.engine.movies_df.iloc[i]["id"]): float(
                max(0.0, min(1.0, similarities[i] * strength))
            )
            for i in range(len(similarities))
        }

    def movie_behavior_boosts(self, bundle: dict) -> dict[int, float]:
        """Direct per-movie ranking nudges from explicit signals.

        These sit in the same +/-BEHAVIOR_BOOST_CAP band the hybrid ranker
        already reserves for feedback_boost, so they nudge rather than
        overpower content similarity. Unlike the profile (which shifts taste
        for *similar* movies), these only touch the movies the signal is about.
        """
        now = datetime.now(timezone.utc)
        boosts: dict[int, float] = {}

        def add(movie_id: int, w: float) -> None:
            boosts[movie_id] = boosts.get(movie_id, 0.0) + w

        for movie_id, rating, ts in bundle.get("ratings", []):
            add(movie_id, (int(round(rating)) - 3) / 2.0 * RATING_NUDGE * signal_decay(ts, now))
        for movie_id, ts in bundle.get("watchlist", []):
            add(movie_id, WATCHLIST_NUDGE * signal_decay(ts, now))

        view_totals: dict[int, float] = {}
        for movie_id, ts in bundle.get("views", []):
            view_totals[movie_id] = view_totals.get(movie_id, 0.0) + VIEW_NUDGE * signal_decay(ts, now)
        for movie_id, total in view_totals.items():
            add(movie_id, min(total, VIEW_NUDGE_CAP))

        for movie_id, fb, ts in bundle.get("feedback", []):
            if fb == "down":
                add(movie_id, FEEDBACK_DOWN_NUDGE * signal_decay(ts, now))
            elif fb == "up":
                add(movie_id, FEEDBACK_UP_NUDGE * signal_decay(ts, now))

        return {
            movie_id: max(-BEHAVIOR_BOOST_CAP, min(BEHAVIOR_BOOST_CAP, b))
            for movie_id, b in boosts.items()
        }

    def has_enough_behavioral_signal(self, bundle: dict) -> bool:
        """True when any signal maps to a movie we can actually build a taste
        profile from (i.e. the user isn't a cold start)."""
        weights = self.aggregate_signal_weights(bundle)
        return any(
            movie_id in self.engine._id_to_index and abs(w) >= MIN_SIGNAL_WEIGHT
            for movie_id, w in weights.items()
        )
