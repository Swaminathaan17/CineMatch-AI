import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_engine.content_similarity import ContentSimilarityEngine, build_soup


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        [
            {
                "id": 1, "title": "Movie A",
                "genres": "Action,Sci-Fi", "cast": "Actor One,Actor Two",
                "director": "Director X", "keywords": "space,heist",
            },
            {
                "id": 2, "title": "Movie B",
                "genres": "Action,Sci-Fi", "cast": "Actor Three,Actor Four",
                "director": "Director X", "keywords": "space,dream",
            },
            {
                "id": 3, "title": "Movie C",
                "genres": "Romance,Comedy", "cast": "Actor Five,Actor Six",
                "director": "Director Y", "keywords": "wedding,love",
            },
        ]
    )


def test_build_soup_keeps_no_stray_apostrophes():
    row = pd.Series({"genres": "Action", "cast": "Christian Bale", "director": "Christopher Nolan", "keywords": "heist"})
    soup = build_soup(row)
    assert "'" not in soup


def test_similar_movies_rank_shared_director_and_genre_higher(sample_df):
    engine = ContentSimilarityEngine(sample_df)
    results = engine.get_similar(1, top_n=2)

    # Movie B shares director + genre with Movie A, Movie C shares nothing -
    # Movie B must rank strictly above Movie C
    ids_in_order = [r["id"] for r in results]
    assert ids_in_order.index(2) < ids_in_order.index(3)


def test_similar_movies_excludes_self(sample_df):
    engine = ContentSimilarityEngine(sample_df)
    results = engine.get_similar(1, top_n=5)
    assert all(r["id"] != 1 for r in results)


def test_shared_fields_readable_not_concatenated(sample_df):
    engine = ContentSimilarityEngine(sample_df)
    results = engine.get_similar(1, top_n=1)
    # "Director X" should stay as one readable string with a space,
    # not "DirectorX" (regression test for the space-stripping bug)
    assert "Director X" in results[0]["shared_director"]


def test_unknown_movie_id_raises(sample_df):
    engine = ContentSimilarityEngine(sample_df)
    with pytest.raises(ValueError):
        engine.get_similar(999)
