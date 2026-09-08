"""Phase 2.4 - AI Discovery intent parsing, mood mapping and bounded scoring.

Covers the full spec: structured intent extraction, query normalization,
reference-movie handling, mood-based ranking boosts, cold-start behaviour,
behavioral-signal composition, bounded boosts, determinism, no raw-search
leakage, Phase 2.3 explanation compatibility, response-shape preservation and
zero per-candidate database queries.
"""
import asyncio
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from ml_engine.discovery_intent import (
    parse_intent,
    normalize_query,
    extract_reference_title,
    extract_keywords,
)
from app.services.ai_discovery_service import AIDiscoveryService
from app.services.recommendation_service import recommendation_service
from app.services.tmdb_client import tmdb_client
from ml_engine.content_similarity import ContentSimilarityEngine
from ml_engine.personalization import PersonalizationEngine
from ml_engine.recommendation_explainer import build_recommendation_explanations


def _intent(text):
    return parse_intent(text)


def _candidate(iid, title, match, source="local", overview="", rating=7.0, pop=50.0):
    return {
        "id": iid, "title": title, "poster_path": None, "release_date": "",
        "overview": overview, "vote_average": rating, "popularity": pop,
        "source": source, "match_score": match, "reference_match": False,
    }


# --- pure parser / normalization -------------------------------------------------

def test_something_funny_detects_comedy_and_mood():
    i = _intent("I want something funny tonight")
    assert "Comedy" in i["genres"]
    assert "Funny" in i["moods"]
    assert "Funny" in i["tone"]


def test_dark_psychological_thriller():
    i = _intent("Give me a dark psychological thriller")
    assert "Thriller" in i["genres"]
    assert "Dark" in i["moods"]
    assert "Dark" in i["tone"]
    assert "Psychological" in i["moods"]


def test_mind_bending_scifi():
    i = _intent("I want a mind-bending sci-fi")
    assert "Science Fiction" in i["genres"]
    assert "Mystery" in i["genres"]
    assert "Mind-bending" in i["moods"]


def test_romantic_and_light():
    i = _intent("Something romantic and light")
    assert "Romance" in i["genres"]
    assert "Romantic" in i["moods"]
    assert "Light" in i["moods"]
    assert "Light" in i["tone"]


def test_something_scary():
    i = _intent("something scary")
    assert "Horror" in i["genres"]
    assert "Scary" in i["moods"]


def test_something_emotional():
    i = _intent("I want an emotional movie")
    assert "Drama" in i["genres"]
    assert "Emotional" in i["moods"]


def test_extracts_reference_title():
    assert _intent("movies like Interstellar")["reference_title"] == "Interstellar"
    assert _intent("something similar to Inception")["reference_title"] == "Inception"
    assert _intent("Give me movies like The Dark Knight")["reference_title"] == "The Dark Knight"
    assert extract_reference_title('something like "Arrival" but shorter') == "Arrival"


def test_synonym_normalization():
    assert "Science Fiction" in _intent("scifi movie")["genres"]
    assert "Science Fiction" in _intent("sci-fi movie")["genres"]
    assert "Science Fiction" in _intent("science fiction movie")["genres"]
    assert normalize_query("sci-fi, feel-good, mind-bending") == "science fiction feel good mind bending"
    rom = _intent("give me a rom-com")
    assert "Romance" in rom["genres"] and "Comedy" in rom["genres"]


def test_punctuation_and_casing_normalization():
    a = _intent("I want a MIND-BENDING SciFi MOVIE!")
    b = _intent("i want a mind bending scifi movie")
    assert a["genres"] == b["genres"]
    assert a["moods"] == b["moods"]
    assert a["reference_title"] == b["reference_title"]


def test_multiple_intents_handled():
    i = _intent("something funny and romantic")
    assert {"Comedy", "Romance"} <= set(i["genres"])
    assert "Funny" in i["moods"] and "Romantic" in i["moods"]


def test_exclusion_keeps_positive_mood():
    i = _intent("something scary but not too gory")
    assert "Horror" in i["genres"]
    assert "Scary" in i["moods"]
    assert i["exclude_genres"] == []


def test_unknown_query_does_not_crash():
    i = _intent("what is this random question about pancakes")
    assert isinstance(i, dict)
    assert i["genres"] == [] and i["moods"] == [] and i["confidence"] == 0.0


def test_empty_query_does_not_crash():
    assert isinstance(_intent(""), dict)
    assert isinstance(_intent("   "), dict)
    assert _intent("")["genres"] == []
    assert _intent("")["confidence"] == 0.0


def test_repeated_parse_is_deterministic():
    a = _intent("give me a dark psychological mind-bending thriller")
    b = _intent("give me a dark psychological mind-bending thriller")
    assert a == b


def test_keywords_are_content_words_only():
    kws = extract_keywords("I want something funny tonight")
    assert "funny" in kws
    assert "want" not in kws and "something" not in kws and "i" not in kws


# --- Phase 2.6 regression: reference titles must never leak mood keyword ---

def test_movies_like_dark_knight_does_not_detect_dark_mood():
    """'movies like The Dark Knight' must resolve the reference title and NOT
    treat 'dark' (a word inside the title) as a mood, tone, or genre hint."""
    i = _intent("movies like The Dark Knight")
    assert i["reference_title"] == "The Dark Knight"
    assert i["moods"] == []
    assert i["tone"] == []
    assert i["genres"] == []  # no Thriller/Crime/Horror invented from "dark"
    assert "dark" not in i["keywords"]


def test_similar_to_dark_knight_does_not_detect_dark_mood():
    i = _intent("similar to The Dark Knight")
    assert i["reference_title"] == "The Dark Knight"
    assert i["moods"] == [] and i["tone"] == [] and i["genres"] == []


def test_dark_psychological_thrillers_still_detects_moods():
    """Legitimate mood-only queries must keep working after reference masking."""
    i = _intent("dark psychological thrillers")
    assert i["reference_title"] is None
    assert "Dark" in i["moods"]
    assert "Psychological" in i["moods"]
    assert "Thriller" in i["genres"]
    assert "Horror" in i["genres"]


def test_quoted_title_with_mood_word_is_reference_not_mood():
    i = _intent('similar to "Dark Waters"')
    assert i["reference_title"] == "Dark Waters"
    assert i["moods"] == [] and i["tone"] == [] and i["genres"] == []


def test_title_plus_actual_mood_outside_title_still_detected():
    i = _intent("movies like The Dark Knight but emotional")
    assert i["reference_title"] == "The Dark Knight"
    assert "Emotional" in i["moods"]
    assert "Drama" in i["genres"]
    assert "Dark" not in i["moods"]


def test_mood_outside_reference_title_still_detected():
    i = _intent("dark movies like Interstellar")
    assert i["reference_title"] == "Interstellar"
    assert "Dark" in i["moods"]


def test_reference_title_words_not_in_keyword_bag():
    i = _intent("movies like The Dark Knight")
    assert "knight" not in i["keywords"] and "dark" not in i["keywords"]


def test_mask_helper_is_deterministic_and_preserves_rest():
    from ml_engine.discovery_intent import _mask_reference_title

    low = "movies like the dark knight but funny"
    first = _mask_reference_title(low, "The Dark Knight")
    second = _mask_reference_title(low, "The Dark Knight")
    assert first == second == "movies like but funny"


# --- ranking / composition -----------------------------------------------------

def _svc_intent():
    return _intent("i want something funny")


def _rank(svc, items, intent, ref_id=None, genre_map=None, tmdb=None):
    return svc._merge_and_rank(
        copy.deepcopy(items),
        tmdb or [],
        intent,
        10,
        ref_id=ref_id,
        genre_map=genre_map or {},
    )


def test_intent_boosts_are_bounded():
    svc = AIDiscoveryService()
    intent = {
        "query": "something funny", "genres": [], "mood_genres": ["Comedy"],
        "moods": ["Funny"], "tone": ["Funny"], "keywords": ["funny"],
        "exclude_genres": [], "sort": "relevance", "runtime_max": None,
    }
    plain = _candidate(1, "The Serious One", 0.5, rating=7.0)
    funny = _candidate(2, "The Comedic One", 0.5, rating=7.0)
    ranked = _rank(svc, [plain, funny], intent, genre_map={1: "Drama", 2: "Comedy"})

    pct = {r["id"]: r["match_percentage"] for r in ranked}
    # identical bases except the mood/genre match -> the mood boost cannot
    # exceed its declared cap (0.10) on its own
    assert abs(pct[2] - pct[1]) <= 10
    for r in ranked:
        assert 1 <= r["match_percentage"] <= 99


def test_behavioral_preference_still_dominates():
    svc = AIDiscoveryService()
    intent = _svc_intent()  # comedy via mood
    strong_behavior = _candidate(10, "Serious Epic", 0.95, rating=7.0)
    weak_but_on_intent = _candidate(11, "Small Comedy", 0.50, rating=7.0)
    ranked = _rank(
        svc, [strong_behavior, weak_but_on_intent], intent,
        genre_map={10: "Drama", 11: "Comedy"},
    )
    # a strong behavioural/content signal (match_score) still outranks an
    # intent-tilted but weak candidate - intent boosts are a layer, not a takeover
    assert ranked[0]["id"] == 10


def test_reference_similarity_is_deterministic():
    svc = AIDiscoveryService()

    class FakeEngine:
        def similarity_between_ids(self, a, b):
            return 0.8

    items = [
        _candidate(1, "Space Odyssey-like A", 0.4),
        _candidate(2, "Something Else", 0.4),
    ]
    intent = _intent("something like Interstellar")
    again = lambda: _rank(svc, items, intent, ref_id=1, genre_map={1: "Sci-fi", 2: "Drama"})
    first = again()
    second = again()
    assert [r["match_percentage"] for r in first] == [r["match_percentage"] for r in second]
    assert [r["id"] for r in first] == [r["id"] for r in second]


def test_no_raw_search_text_in_reasons(monkeypatch):
    svc = AIDiscoveryService()
    intent = _intent("give me a mind-bending sci-fi with a clever twist")
    item = _candidate(1, "The Clever Flick", 0.5, overview="a mind bending science fiction mystery")
    ranked = _rank(svc, [item], intent, genre_map={1: "Science Fiction, Mystery"})
    reasons = ranked[0]["reasons"]
    assert "give me a mind-bending" not in " ".join(reasons).lower()
    assert all("your request" in r.lower() or "genre" in r.lower() or "reference movie" in r.lower() or
               "highly rated" in r.lower() or "Matched" in r or "Found" in r for r in reasons)


def test_no_per_candidate_db_query(monkeypatch):
    svc = AIDiscoveryService()
    calls = {"db": 0}

    def fake_get_db():  # pragma: no cover - only reached if code hits the DB
        calls["db"] += 1
        raise AssertionError("unexpected DB access")

    import app.db.session as db_session
    monkeypatch.setattr(db_session, "get_db", fake_get_db)

    intent = _intent("something like Interstellar")

    class FakeEngine:
        sim_calls = 0

        def similarity_between_ids(self, a, b):
            self.sim_calls += 1
            return 0.6

    fake = FakeEngine()
    monkeypatch.setattr(recommendation_service, "_engine", fake)
    items = [_candidate(1, "Local A", 0.5), _candidate(2, "Local B", 0.5), _candidate(3, "Local C", 0.5)]
    _rank(svc, items, intent, ref_id=1, genre_map={1: "Sci-fi", 2: "Drama", 3: "Thriller"})
    assert calls["db"] == 0
    assert fake.sim_calls <= len(items)  # one bounded call per candidate, never N^2


# --- async discover integration (cold start / response shape) -----------------

def _fake_df():
    return pd.DataFrame([
        {"id": 1, "title": "The Big Laugh", "genres": "Comedy", "overview": "a hilarious comedy about a party",
         "poster_path": None, "release_date": "2020-01-01", "vote_average": 7.4, "popularity": 60.0, "source": "local"},
        {"id": 2, "title": "Deep Dread", "genres": "Horror, Thriller", "overview": "a terrifying thriller in a dark house",
         "poster_path": None, "release_date": "2019-06-01", "vote_average": 7.8, "popularity": 75.0, "source": "local"},
        {"id": 3, "title": "Mind Maze", "genres": "Science Fiction, Mystery", "overview": "a puzzling sci-fi mystery about dreams",
         "poster_path": None, "release_date": "2021-11-01", "vote_average": 8.2, "popularity": 90.0, "source": "local"},
    ])


def _patch_discovery(monkeypatch):
    svc = AIDiscoveryService()
    df = _fake_df()
    monkeypatch.setattr(recommendation_service, "_df", df)

    def fake_search_by_mood(query, top_n=10):
        return {
            "mode": "fallback",
            "interpreted_genres": ["Comedy"],
            "results": [
                {"id": int(r["id"]), "title": r["title"], "match_score": 0.55 - i * 0.05}
                for i, (_, r) in enumerate(df.iterrows())
            ],
        }

    monkeypatch.setattr(recommendation_service, "search_by_mood", fake_search_by_mood)
    monkeypatch.setattr(recommendation_service, "search_local", lambda q, top_n=5: [])

    async def _empty(_params=None):
        return {"results": []}

    monkeypatch.setattr(tmdb_client, "search_movies", _empty)
    monkeypatch.setattr(tmdb_client, "discover_movies", _empty)
    monkeypatch.setattr(tmdb_client, "get_recommendations", _empty)
    return svc


def test_cold_start_intent_based_recommendations(monkeypatch):
    svc = _patch_discovery(monkeypatch)
    res = asyncio.run(svc.discover("I want something funny tonight", top_n=5))
    assert len(res["results"]) > 0
    assert "Comedy" in res["intent"]["genres"]
    assert "Funny" in res["intent"]["moods"]
    assert res["results"][0]["id"] == 1  # Comedy movie tops a cold-start funny request


def test_api_response_fields_remain_intact(monkeypatch):
    svc = _patch_discovery(monkeypatch)
    res = asyncio.run(svc.discover("something scary but not too gory", top_n=5))
    assert set(res.keys()) >= {"query", "intent", "mode", "sources", "results", "explanation"}
    assert set(res["sources"].keys()) >= {"local", "tmdb", "tmdb_mode"}
    assert set(res["explanation"].keys()) >= {"interpreted_as", "semantic_engine", "tmdb_used"}
    for r in res["results"]:
        assert set(r.keys()) >= {
            "id", "title", "poster_path", "release_date", "vote_average",
            "popularity", "source", "match_score", "match_percentage", "reasons",
        }
        assert "id" in r and "title" in r
    # normalized query is internal only - never leaked to clients
    assert "normalized_query" not in res["intent"]


# --- Phase 2.3 explanation compatibility ---------------------------------------

def test_phase3_explanations_remain_valid():
    df = pd.DataFrame([
        {"id": 1, "title": "Deep Space", "overview": "astronaut survives alone in space",
         "genres": "Science Fiction, Drama", "cast": "A", "director": "Nolan", "keywords": "space, survival"},
        {"id": 2, "title": "Mars Survival", "overview": "astronaut fights to survive alone on mars",
         "genres": "Science Fiction, Drama", "cast": "B", "director": "Scott", "keywords": "mars, survival"},
    ])
    engine = ContentSimilarityEngine(df)
    personal = PersonalizationEngine(engine)
    discovery_results = [
        {"id": 2, "title": "Mars Survival", "match_percentage": 95, "reasons": ["Found on TMDB"]},
    ]
    expls = build_recommendation_explanations(discovery_results, personal, signals={})
    assert len(expls) == 1
    assert set(expls[0].keys()) == {"primary", "secondary"}
    assert isinstance(expls[0]["primary"], str) and expls[0]["primary"]
    assert expls[0]["secondary"] is None or isinstance(expls[0]["secondary"], str)
    assert discovery_results[0]["match_percentage"] == 95  # untouched


def test_parse_intent_reference_then_mood_combined():
    i = _intent("movies like Interstellar but dark and emotional")
    assert i["reference_title"] == "Interstellar"
    assert "Dark" in i["moods"]
    assert "Emotional" in i["moods"]
    assert i["confidence"] >= 0.6