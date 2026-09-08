import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_engine.discovery_intent import parse_intent

def test_reference_and_constraints():
    x = parse_intent("Something like Interstellar but darker and more emotional, under 150 minutes")
    assert x["reference_title"] == "Interstellar"
    assert "Thriller" in x["genres"] or "Drama" in x["genres"]
    assert x["runtime_max"] == 150

def test_recent_scifi():
    x = parse_intent("recent sci-fi movies about survival in space")
    assert "Science Fiction" in x["genres"]
    assert x["sort"] == "recent"
    assert "space" in x["themes"]
    assert "survival" in x["themes"]

def test_rating_and_year():
    x = parse_intent("highly rated thriller movies from 2019")
    assert "Thriller" in x["genres"]
    assert x["year"] == 2019
    assert x["min_rating"] >= 7
