import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from evaluation.benchmark import ndcg_at_k, precision_at_k, run_benchmark


def test_metric_helpers_are_bounded():
    rels = [3, 2, 0, 1]
    assert 0 <= precision_at_k(rels, 3) <= 1
    assert 0 <= ndcg_at_k(rels, 4) <= 1


def test_benchmark_runs_on_small_fixture(tmp_path):
    df = pd.DataFrame([
        {"id": 1, "title": "A", "overview": "space survival", "genres": "Science Fiction, Drama", "cast": "X,Y", "director": "D1", "keywords": "space, survival", "vote_average": 8, "popularity": 10},
        {"id": 2, "title": "B", "overview": "space survival", "genres": "Science Fiction, Drama", "cast": "X,Z", "director": "D1", "keywords": "space, survival", "vote_average": 8, "popularity": 9},
        {"id": 3, "title": "C", "overview": "romantic wedding", "genres": "Romance", "cast": "Q", "director": "D2", "keywords": "wedding", "vote_average": 7, "popularity": 8},
        {"id": 4, "title": "D", "overview": "space adventure", "genres": "Science Fiction", "cast": "M", "director": "D3", "keywords": "space", "vote_average": 7, "popularity": 7},
        {"id": 5, "title": "E", "overview": "crime drama", "genres": "Crime, Drama", "cast": "N", "director": "D4", "keywords": "crime", "vote_average": 7, "popularity": 6},
    ])
    path = tmp_path / "movies.csv"
    df.to_csv(path, index=False)
    result = run_benchmark(path, seeds=[1,2,3,4,5], k=3)
    assert result["benchmark"]["seed_count"] == 5
    assert "ndcg" in result["metrics"]
