import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from ml_engine.nl_query_parser import NLQueryEngine


@pytest.fixture(scope="module")
def engine():
    df = pd.DataFrame(
        [
            {"id": 1, "title": "Reindeer Games", "genres": "Thriller,Crime",
             "overview": "A con man is coerced into a violent heist with a deadly double-cross ending."},
            {"id": 2, "title": "Disaster Movie", "genres": "Action,Comedy",
             "overview": "A spoof comedy parodying big-budget disaster films with slapstick gags."},
            {"id": 3, "title": "Mulholland Drive", "genres": "Mystery,Sci-Fi",
             "overview": "An amnesiac actress and a woman navigate a surreal blur of dreams and reality."},
            {"id": 4, "title": "Boxing Helena", "genres": "Drama,Romance",
             "overview": "A surgeon becomes obsessed with a woman after a tragic accident."},
        ]
    )
    return NLQueryEngine(df)


def test_genre_boost_deprioritizes_unrelated_comedy_for_heist_query(engine):
    """Regression test for a real failure found during manual testing:
    'a heist movie with a twist' incorrectly ranked 'Disaster Movie' (a
    spoof comedy sharing zero real thematic content) above genuinely
    relevant thriller results, purely from noisy TF-IDF overlap."""
    result = engine.search("a heist movie with a clever twist ending", top_n=4)
    titles_in_order = [r["title"] for r in result["results"]]
    assert titles_in_order.index("Reindeer Games") < titles_in_order.index("Disaster Movie")


def test_genre_boost_deprioritizes_unrelated_drama_for_scifi_query(engine):
    result = engine.search("a mind-bending sci-fi movie about dreams and reality", top_n=4)
    titles_in_order = [r["title"] for r in result["results"]]
    assert titles_in_order.index("Mulholland Drive") < titles_in_order.index("Boxing Helena")


def test_literal_genre_mention_detected_without_mood_adjective():
    """Regression test: a query with no mood adjective at all (just literal
    content words like 'astronauts', 'mars') previously got zero interpreted
    genres and fell back to pure noisy TF-IDF. Also guards against the naming
    mismatch bug where the hint dict said 'Sci-Fi' but the real dataset uses
    'Science Fiction' - the hint was silently never matching anything."""
    from ml_engine.nl_query_parser import extract_mood_hints

    hints = extract_mood_hints("astronauts stranded on mars trying to survive")
    assert "Science Fiction" in hints


def test_sci_fi_hint_matches_dataset_genre_string_exactly():
    """The dataset's real genre string is 'Science Fiction', not 'Sci-Fi' -
    every hint that should trigger on sci-fi content must produce the former,
    or the genre-boost's set intersection silently never fires."""
    from ml_engine.nl_query_parser import MOOD_GENRE_HINTS

    all_hinted_genres = {g for genres in MOOD_GENRE_HINTS.values() for g in genres}
    assert "Sci-Fi" not in all_hinted_genres
