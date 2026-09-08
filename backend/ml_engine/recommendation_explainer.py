"""Transparent "why this movie?" explanations for personalized recommendations.

Deterministic, pure, side-effect-free. Consumes the already-batched Phase 2.2
signal bundle plus the per-candidate ranking context (behavior weights, direct
boosts, preference scores) and the shared ContentSimilarityEngine for bounded
pairwise checks against a movie's strongest signals. Never touches the
database and never refits a vectorizer.

Every claim maps 1:1 to a signal that actually moved the Phase 2.2 score:

- similarity-based reasons ("Because you liked X") are gated on the same
  aggregate weights/preference scores the ranker used and require a genuine
  content-similarity floor between the candidate and that signal movie.
- genre/recency/text reasons use real movie metadata, timestamps inside the
  recency window, and only the genre hints already extracted by Phase 2.2's
  search logic - raw search text is never exposed.
- negative feedback can only suppress; it can never produce a positive claim.

Output shape: {"primary": str, "secondary": str | None}. Never empty - a
truthful fallback is always provided.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ml_engine.personalization import (
    MIN_SIGNAL_WEIGHT,
    PersonalizationEngine,
    SIGNAL_HALF_LIFE_DAYS,
    _as_utc,
)

# Similarity floors that must be met before claiming "you liked/rated/
# watchlisted something like this". Below these the candidate is not genuinely
# similar, so claiming causality would be fabrication.
LIKE_SIM_THRESHOLD = 0.25
RATED_SIM_THRESHOLD = 0.25
WATCHLIST_SIM_THRESHOLD = 0.25

# Per-category cap on how many signal movies we compare a candidate against.
# Keeps per-request pairwise work bounded and O(1) in dataset size.
MAX_SIGNAL_MOVIES_PER_CATEGORY = 5

# "Because you often watch X" requires the genre to sum at least this much
# positive signal mass (a single like = 1.0, a single view = ~0.15).
GENRE_STRENGTH_FLOOR = 0.5

# "Because you've been watching more X lately" - only views inside the standard
# Phase 2.2 recency window (the signal half-life) qualify as "lately".
RECENT_VIEW_WINDOW_DAYS = SIGNAL_HALF_LIFE_DAYS

# Deterministic priority order. Lower wins.
RANKS = {
    "liked_similar": 1,
    "rated_self": 2,
    "rated_similar": 3,
    "watchlist_self": 4,
    "genre": 5,
    "watchlist_similar": 6,
    "viewing": 7,
    "search": 8,
    "feedback_up": 9,
}


def _clean_genres(value) -> list[str]:
    if isinstance(value, list):
        return [g for g in value if g]
    if isinstance(value, str):
        return [g.strip() for g in value.split(",") if g.strip()]
    return []


def _similarity_matrix(engine, candidate_ids: list[int], signal_ids: list[int]):
    """Vectorized candidate x signal content similarity (0.72 meta / 0.28
    overview blend - the same composition the ranker uses). No vectorizer is
    refit; this is a bounded matmul over the already-built sparse matrices.
    """
    cand_idx = [i for i in candidate_ids if i in engine._id_to_index]
    sig_idx = [i for i in signal_ids if i in engine._id_to_index]
    if not cand_idx or not sig_idx:
        return None, [], []
    cand_pos = [engine._id_to_index[i] for i in cand_idx]
    sig_pos = [engine._id_to_index[i] for i in sig_idx]
    meta = cosine_similarity(engine.meta_matrix[cand_pos], engine.meta_matrix[sig_pos])
    if engine._has_overview_vocab:
        overview = cosine_similarity(
            engine.overview_matrix[cand_pos], engine.overview_matrix[sig_pos]
        )
    else:
        overview = np.zeros((len(cand_pos), len(sig_pos)))
    return 0.72 * meta + 0.28 * overview, cand_idx, sig_idx


def build_recommendation_explanations(
    results: list[dict],
    personal: PersonalizationEngine,
    signals: dict,
    weights: dict | None = None,
    preference_scores: dict | None = None,
    behavior_boosts: dict | None = None,
    mode: str = "personalized",
    now: datetime | None = None,
) -> list[dict]:
    """Return one {"primary", "secondary"} explanation per result, in order.

    `preference_scores`/`behavior_boosts`/`weights` are expected to already be
    computed by the caller (as the ranking path does) so nothing is recomputed
    per request. If omitted they are derived here for standalone use.
    """
    now = now or datetime.now(timezone.utc)
    engine = personal.engine

    if weights is None:
        weights = personal.aggregate_signal_weights(signals)
    if preference_scores is None:
        preference_scores = personal.behavior_scores(signals)
    if behavior_boosts is None:
        behavior_boosts = personal.movie_behavior_boosts(signals)

    candidate_ids = [int(r["id"]) for r in results if "id" in r]
    if not candidate_ids:
        return []

    rows = {mid: engine.movies_df.iloc[engine._id_to_index[mid]]
            for mid in candidate_ids if mid in engine._id_to_index}
    title_of = lambda mid: str(engine.movies_df.iloc[engine._id_to_index[mid]]["title"])

    def contributing(ids) -> list[int]:
        return sorted(
            {
                mid
                for mid in ids
                if mid in engine._id_to_index
                and abs(weights.get(mid, 0.0)) >= MIN_SIGNAL_WEIGHT
            },
            key=lambda mid: (-abs(weights.get(mid, 0.0)), mid),
        )[:MAX_SIGNAL_MOVIES_PER_CATEGORY]

    ratings = {mid: r for mid, r, _ in signals.get("ratings", [])}
    liked_pool = contributing(signals.get("liked_ids", []) or [m for m, _ in signals.get("liked", [])])
    rated_hi_pool = contributing([mid for mid, r in ratings.items() if r >= 4 and mid not in candidate_ids])
    watch_pool = contributing(signals.get("watchlist_ids", []) or [m for m, _ in signals.get("watchlist", [])])
    viewed_recent_ids = [
        mid
        for mid, ts in signals.get("views", [])
        if _as_utc(ts) is not None and (now - _as_utc(ts)).total_seconds() <= RECENT_VIEW_WINDOW_DAYS * 86400
    ]
    viewed_pool = contributing(viewed_recent_ids)
    feedback_up_ids = {mid for mid, fb, _ in signals.get("feedback", []) if fb == "up"}

    # Search hints are query-derived and identical for every candidate - compute
    # once instead of re-extracting them inside the per-candidate loop.
    search_hints = set(personal.search_genre_hints(signals))

    # One bounded similarity matrix for the whole candidate list.
    signal_union = sorted({*liked_pool, *rated_hi_pool, *watch_pool, *viewed_pool})
    sim_matrix, cand_ids, sig_ids = _similarity_matrix(engine, candidate_ids, signal_union)
    sims = {}
    if sim_matrix is not None:
        cpos = {mid: i for i, mid in enumerate(cand_ids)}
        spos = {mid: i for i, mid in enumerate(sig_ids)}
        for ci, cid in enumerate(cand_ids):
            for si, sid in enumerate(sig_ids):
                sims[(cid, sid)] = float(sim_matrix[ci, si])

    def sim(cid: int, sid: int) -> float:
        return sims.get((cid, sid), 0.0)

    # Genres whose positive signal mass is large enough to call a preference.
    # "Often watch" category is deliberately fed only by like/rating/view mass -
    # adding to a watchlist is exploration, not a viewing habit, so a movie with
    # only a watchlist signal must not surface as "you often watch X".
    gaze_ids = (
        set(liked_pool)
        | set(ratings.keys())
        | set(signals.get("viewed_ids", []) or [m for m, _ in signals.get("views", [])])
    )
    genre_strength: dict[str, float] = {}
    for mid, w in weights.items():
        if w <= 0 or mid not in engine._id_to_index or mid not in gaze_ids:
            continue
        for g in _clean_genres(engine.movies_df.iloc[engine._id_to_index[mid]].get("genres")):
            genre_strength[g] = genre_strength.get(g, 0.0) + w

    def reason_for(cid: int) -> list[dict]:
        row = rows.get(cid)
        if row is None:
            return []
        cand_genres = _clean_genres(row.get("genres"))
        pref = preference_scores.get(cid, 0.0)
        rs = []

        # 1 - strongly similar to a movie the user liked
        if liked_pool and pref > 0:
            best = max(liked_pool, key=lambda sid: sim(cid, sid))
            s = sim(cid, best)
            if s >= LIKE_SIM_THRESHOLD:
                rs.append({"category": "liked_similar", "strength": s,
                           "text": f"Because you liked {title_of(best)}"})

        # 2 - the user rated this exact movie 4-5 stars
        if cid in ratings and int(round(ratings[cid])) >= 4 and abs(weights.get(cid, 0.0)) >= MIN_SIGNAL_WEIGHT:
            rs.append({"category": "rated_self",
                       "strength": abs(weights.get(cid, 0.0)),
                       "text": f"Because you rated {title_of(cid)} {int(round(ratings[cid]))}★"})

        # 3 - strongly similar to a movie the user rated 4-5 stars
        if rated_hi_pool and pref > 0:
            best = max(rated_hi_pool, key=lambda sid: sim(cid, sid))
            s = sim(cid, best)
            if s >= RATED_SIM_THRESHOLD:
                rs.append({"category": "rated_similar", "strength": s,
                           "text": f"Because you rated {title_of(best)} {int(round(ratings[best]))}★"})

        # 4 - the user added this exact movie to their watchlist
        if cid in watch_pool and weights.get(cid, 0.0) > 0:
            rs.append({"category": "watchlist_self", "strength": weights[cid],
                       "text": f"Because you added {title_of(cid)} to your watchlist"})

        # 5 - strong genre preference learned from positive signals
        if pref > 0 and cand_genres:
            strong = [g for g in cand_genres if genre_strength.get(g, 0.0) >= GENRE_STRENGTH_FLOOR]
            if strong:
                g = max(strong, key=lambda g: (genre_strength[g], g))
                rs.append({"category": "genre", "strength": genre_strength[g],
                           "text": f"Because you often watch {g}"})

        # 6 - similar to watchlist additions
        if watch_pool and pref > 0:
            others = [sid for sid in watch_pool if sid != cid]
            if others:
                best = max(others, key=lambda sid: sim(cid, sid))
                s = sim(cid, best)
                if s >= WATCHLIST_SIM_THRESHOLD:
                    rs.append({"category": "watchlist_similar", "strength": s,
                               "text": "Because you've added similar movies to your watchlist"})

        # 7 - viewing preference from movies actually viewed in the recency window
        if pref > 0 and cand_genres and viewed_pool:
            viewed_genre = {}
            for vid in viewed_pool:
                for g in _clean_genres(engine.movies_df.iloc[engine._id_to_index[vid]].get("genres")):
                    viewed_genre[g] = viewed_genre.get(g, 0.0) + abs(weights.get(vid, 0.0))
            match = [g for g in cand_genres if viewed_genre.get(g, 0.0) >= MIN_SIGNAL_WEIGHT]
            if match:
                g = max(match, key=lambda g: (viewed_genre[g], g))
                rs.append({"category": "viewing", "strength": viewed_genre[g],
                           "text": f"Because you've been watching more {g} lately"})

        # 8 - genre hint from a recent search (never the raw query text)
        if pref > 0 and cand_genres:
            match = [g for g in cand_genres if g in search_hints]
            if match:
                g = sorted(match)[0]
                rs.append({"category": "search", "strength": 1.0,
                           "text": f"Because you recently searched for {g}"})

        # 9 - the user marked a recommendation of this movie as good
        if cid in feedback_up_ids:
            rs.append({"category": "feedback_up", "strength": abs(weights.get(cid, 0.0)) or 0.3,
                       "text": "Because you've enjoyed similar recommendations"})

        for r in rs:
            r["rank"] = RANKS[r["category"]]
        return rs

    explanations = []
    for cid in candidate_ids:
        rs = reason_for(cid)
        if not rs:
            if mode == "trending":
                explanations.append({"primary": "Popular right now", "secondary": None})
            else:
                explanations.append({"primary": "Recommended based on your preferences", "secondary": None})
            continue

        rs.sort(key=lambda r: (r["rank"], -r["strength"], r["text"]))
        primary = rs[0]
        secondary = next(
            (r for r in rs[1:] if r["category"] != primary["category"]), None
        )
        explanations.append({
            "primary": primary["text"],
            "secondary": secondary["text"] if secondary else None,
        })

    return explanations