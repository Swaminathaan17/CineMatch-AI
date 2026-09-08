from __future__ import annotations

from app.services.tmdb_client import tmdb_client, TMDBError
from app.services.recommendation_service import recommendation_service
from ml_engine.discovery_intent import parse_intent


class AIDiscoveryService:
    async def discover(self, query: str, top_n: int = 12) -> dict:
        intent = parse_intent(query)
        local = recommendation_service.search_by_mood(query, top_n=max(top_n * 3, 30))
        local_rows = []
        local_df = recommendation_service._df
        genre_map = {}
        if local_df is not None:
            by_id = {int(r["id"]): r for _, r in local_df.iterrows()}
            for item in local["results"]:
                r = by_id.get(int(item["id"]))
                if r is not None:
                    local_rows.append(self._result_from_row(r, item.get("match_score", 0), "local"))
                    genres = str(r.get("genres") or "")
                    if genres:
                        genre_map[int(r["id"])] = genres

        # Resolve a "something like X" reference to a local movie so the shared
        # content-similarity vectors can boost genuinely similar candidates.
        # Silently skips if the title can't be resolved locally; TMDB reference
        # enrichment below is attempted independently.
        ref_id = None
        if intent["reference_title"]:
            try:
                for hit in recommendation_service.search_local(intent["reference_title"], top_n=5) or []:
                    if hit.get("id"):
                        ref_id = int(hit["id"])
                        break
            except Exception:
                ref_id = None

        tmdb_results = []
        tmdb_source = "disabled"
        if intent["reference_title"]:
            tmdb_results = await self._reference_movie(intent["reference_title"], top_n)
            tmdb_source = "reference"
        elif (intent["tmdb_genre_ids"] or intent["genres"] or intent["moods"] or intent["year"]
                or intent["decade"] or intent["runtime_min"] or intent["runtime_max"]
                or intent["min_rating"] or intent["sort"] != "relevance"):
            tmdb_results = await self._discover_tmdb(intent, top_n * 2)
            tmdb_source = "discover"
        elif self._looks_like_title(query):
            tmdb_results = await self._title_search(query, top_n)
            tmdb_source = "title"

        merged = self._merge_and_rank(local_rows, tmdb_results, intent, top_n, ref_id=ref_id, genre_map=genre_map)
        # The normalized query is used internally for matching but is not exposed
        # to clients - only the already-echoed original query stays in the payload.
        public_intent = {k: v for k, v in intent.items() if k != "normalized_query"}
        return {
            "query": query,
            "intent": public_intent,
            "mode": local.get("mode", "fallback"),
            "sources": {"local": bool(local_rows), "tmdb": bool(tmdb_results), "tmdb_mode": tmdb_source},
            "results": merged,
            "explanation": self._explanation(intent, local.get("mode", "fallback"), bool(tmdb_results)),
        }

    @staticmethod
    def _looks_like_title(query: str) -> bool:
        q = query.lower().strip()
        return len(q.split()) <= 8 and not any(x in q for x in ["i want", "give me", "something", "movie with", "movie about", "film with", "in the mood", "recommend"])

    async def _title_search(self, query, top_n):
        try:
            data = await tmdb_client.search_movies(query)
        except TMDBError:
            return []
        return [self._tmdb_result(r, 0.85 - i * 0.03) for i, r in enumerate(data.get("results", [])[:top_n])]

    async def _reference_movie(self, title, top_n):
        try:
            data = await tmdb_client.search_movies(title)
            hits = data.get("results", [])
            if not hits: return []
            ref = hits[0]
            recs = await tmdb_client.get_recommendations(int(ref["id"]))
            return [self._tmdb_result(r, 0.9 - i * 0.01, reference=True) for i, r in enumerate(recs.get("results", [])[:top_n * 2])]
        except TMDBError:
            return []

    async def _discover_tmdb(self, intent, limit):
        params = {"page": 1, "include_adult": False, "language": "en-US", "sort_by": "popularity.desc"}
        ids = intent["tmdb_genre_ids"]
        if ids: params["with_genres"] = "|".join(map(str, ids))
        if intent["min_rating"] is not None: params["vote_average.gte"] = intent["min_rating"]
        params["vote_count.gte"] = 50
        if intent["year"]:
            params["primary_release_year"] = intent["year"]
        elif intent["decade"]:
            params["primary_release_date.gte"] = f"{intent['decade']}-01-01"
            params["primary_release_date.lte"] = f"{intent['decade'] + 9}-12-31"
        if intent["runtime_min"] is not None: params["with_runtime.gte"] = intent["runtime_min"]
        if intent["runtime_max"] is not None: params["with_runtime.lte"] = intent["runtime_max"]
        if intent["sort"] == "recent": params["sort_by"] = "primary_release_date.desc"
        elif intent["sort"] == "rating": params["sort_by"] = "vote_average.desc"
        elif intent["sort"] == "classic": params["sort_by"] = "vote_count.desc"
        try:
            data = await tmdb_client.discover_movies(params)
        except TMDBError:
            return []
        return [self._tmdb_result(r, 0.7, reference=False) for r in data.get("results", [])[:limit]]

    @staticmethod
    def _tmdb_result(r, score=0.5, reference=False):
        return {
            "id": int(r["id"]), "title": r.get("title", ""), "poster_path": r.get("poster_path"),
            "release_date": r.get("release_date", ""), "overview": r.get("overview", ""),
            "vote_average": r.get("vote_average", 0), "popularity": r.get("popularity", 0),
            "source": "tmdb", "match_score": score, "reference_match": reference,
        }

    @staticmethod
    def _result_from_row(r, score, source):
        return {
            "id": int(r["id"]), "title": r["title"], "poster_path": r.get("poster_path"),
            "release_date": r.get("release_date", ""), "overview": r.get("overview", ""),
            "vote_average": float(r.get("vote_average") or 0), "popularity": float(r.get("popularity") or 0),
            "source": source, "match_score": float(score), "reference_match": False,
        }

    def _merge_and_rank(self, local, tmdb, intent, top_n, ref_id=None, genre_map=None):
        """Merge local + TMDB candidates and apply the bounded intent-based
        scoring layer on top of the existing feature score.

        final = 0.58*match + 0.15*word_overlap
              + 0.12*genre_bonus + 0.10*mood_bonus
              + 0.06*keyword_bonus + 0.15*reference_similarity
              + 0.12*rating + 0.05*popularity - 0.35*exclusion_penalty

        Every intent boost is clamped so a natural-language query can tilt but
        never destroy the underlying content ranking.
        """
        seen = set()
        all_items = []
        for item in local + tmdb:
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            all_items.append(item)
        if not all_items:
            return []

        # Reference-title words are not independent query content - exclude
        # them from the word-overlap bag so ranking a "movies like The Dark
        # Knight" request doesn't tilt toward "dark"/"knight" keywords.
        qwords = set(w.lower() for w in intent["query"].split() if len(w) > 2)
        ref_words = {w.lower() for w in (intent.get("reference_title") or "").split() if len(w) > 2}
        qwords -= ref_words
        wanted = {g.lower() for g in intent["genres"]}
        mood_wanted = {g.lower() for g in intent.get("mood_genres", []) if g.lower() not in wanted}
        mood_labels = intent.get("moods", [])
        excluded = {g.lower() for g in intent["exclude_genres"]}
        intent_keywords = [k.lower() for k in intent.get("keywords", [])]
        engine = recommendation_service._engine
        genre_map = genre_map or {}

        def genres_text(mid):
            return (genre_map.get(mid, "") or "").lower()

        for x in all_items:
            text = f"{x['title']} {x.get('overview', '')}".lower()
            content = f"{text} {genres_text(x['id'])}".strip()
            word_overlap = sum(1 for w in qwords if w in content) / max(1, len(qwords))

            wanted_genres_found = [g for g in wanted if g in content]

            genre_bonus = 0.12 if wanted_genres_found else 0.0
            mood_bonus = 0.0
            if mood_labels and mood_wanted:
                mood_bonus = 0.10 if any(g in content for g in mood_wanted) else 0.0

            keyword_bonus = min(
                0.06,
                0.02 * sum(1 for k in intent_keywords if k in content),
            )
            exclusion_penalty = 0.35 if excluded and any(g in content for g in excluded) else 0.0

            reference_bonus = 0.0
            if ref_id and x.get("source") == "local" and engine is not None:
                try:
                    sim = float(engine.similarity_between_ids(ref_id, int(x["id"])))
                except Exception:
                    sim = 0.0
                if sim >= 0.25:
                    reference_bonus = min(0.15, sim * 0.15)

            rating = min(1, float(x.get("vote_average") or 0) / 10)
            pop = min(1, float(x.get("popularity") or 0) / 100)
            x["final_score"] = max(
                0,
                float(x.get("match_score", 0)) * 0.58
                + word_overlap * 0.15
                + genre_bonus
                + mood_bonus
                + keyword_bonus
                + reference_bonus
                + rating * 0.12
                + pop * 0.05
                - exclusion_penalty,
            )

            reasons = []
            if reference_bonus > 0:
                reasons.append("Similar to your reference movie")
            if x.get("reference_match"):
                reasons.append("Recommended by TMDB from your reference movie")
            if genre_bonus:
                reasons.append("Matches the genres in your request")
            if mood_bonus and mood_labels:
                label = str(mood_labels[0]).title()
                reasons.append(f"Matches your request for something {label}")
            if x["vote_average"] >= 7.5:
                reasons.append("Highly rated by audiences")
            if intent["sort"] == "recent":
                reasons.append("Prioritised for recent releases")
            if intent["runtime_max"]:
                reasons.append("Fits your runtime preference")
            if x["source"] == "local":
                reasons.append("Matched against your recommendation library")
            else:
                reasons.append("Found live on TMDB")

            x["match_percentage"] = min(99, max(1, round(x["final_score"] * 100)))
            x["reasons"] = reasons[:4]
            x.pop("overview", None)
            x.pop("final_score", None)
            x.pop("reference_match", None)

        all_items.sort(key=lambda x: x["match_percentage"], reverse=True)
        return all_items[:top_n]

    @staticmethod
    def _explanation(intent, mode, tmdb_used):
        bits = []
        if intent["genres"]:
            bits.append("genres: " + ", ".join(intent["genres"]))
        if intent["moods"]:
            bits.append("mood: " + ", ".join(intent["moods"]))
        if intent["tone"]:
            bits.append("tone: " + ", ".join(intent["tone"]))
        if intent["themes"]:
            bits.append("themes: " + ", ".join(intent["themes"]))
        if intent["year"]:
            bits.append(f"year: {intent['year']}")
        if intent["runtime_max"]:
            bits.append(f"under {intent['runtime_max']} min")
        if intent["reference_title"]:
            bits.append(f"similar to {intent['reference_title']}")
        return {
            "interpreted_as": bits or ["free-form movie preference"],
            "semantic_engine": mode,
            "tmdb_used": tmdb_used,
        }


ai_discovery_service = AIDiscoveryService()