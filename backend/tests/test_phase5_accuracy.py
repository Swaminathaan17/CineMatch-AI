import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from ml_engine.content_similarity import ContentSimilarityEngine
from ml_engine.personalization import PersonalizationEngine


def make_df():
    return pd.DataFrame([
        {"id": 1, "title": "Deep Space", "overview": "astronaut survives alone in space", "genres": "Science Fiction, Drama", "cast": "A", "director": "Nolan", "keywords": "space, survival"},
        {"id": 2, "title": "Mars Survival", "overview": "astronaut fights to survive alone on mars", "genres": "Science Fiction, Drama", "cast": "B", "director": "Scott", "keywords": "mars, survival"},
        {"id": 3, "title": "Space Comedy", "overview": "crew has a funny adventure in space", "genres": "Science Fiction, Comedy", "cast": "C", "director": "Lee", "keywords": "space, comedy"},
        {"id": 4, "title": "Romantic Wedding", "overview": "friends prepare for a romantic wedding", "genres": "Romance", "cast": "D", "director": "Smith", "keywords": "wedding, love"},
    ])


def test_negative_feedback_moves_profile_away_from_disliked_movie():
    engine = ContentSimilarityEngine(make_df())
    personal = PersonalizationEngine(engine)
    without_negative = personal.preference_scores([1])[2]
    with_negative = personal.preference_scores([1], negative_movie_ids=[3])[2]
    assert with_negative <= without_negative


def test_content_engine_is_not_replaced_by_metadata_only_similarity():
    engine = ContentSimilarityEngine(make_df())
    results = engine.get_similar(1, top_n=3)
    ids = [r["id"] for r in results]
    assert ids[0] == 2
