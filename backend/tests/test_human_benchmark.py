import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from evaluation.human_benchmark import create_annotation_set, evaluate_annotations


def test_human_annotation_set_is_blank_and_reproducible(tmp_path):
    rows=[]
    for i in range(1,13):
        rows.append({"id":i,"title":f"M{i}","overview":"space adventure" if i%2==0 else "romantic drama","genres":"Science Fiction,Adventure" if i%2==0 else "Romance,Drama","cast":"A,B" if i%2==0 else "C,D","director":"D1" if i%2==0 else "D2","keywords":"space,adventure" if i%2==0 else "romance","vote_average":7+i%3,"popularity":10+i})
    path=tmp_path/"movies.csv"; pd.DataFrame(rows).to_csv(path,index=False)
    out=tmp_path/"ratings.csv"
    create_annotation_set(path,out,seed_count=5,k=3)
    result=pd.read_csv(out)
    assert len(result)>0
    assert result["rating"].isna().all()
    assert set(result["current_rank"].dropna().astype(int)).issubset({1,2,3})


def test_human_evaluator_rejects_unrated(tmp_path):
    p=tmp_path/"ratings.csv"
    pd.DataFrame([{"seed_id":1,"candidate_id":2,"current_rank":1,"baseline_rank":1,"rating":"","annotator":"","notes":""}]).to_csv(p,index=False)
    try:
        evaluate_annotations(p)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "No human ratings" in str(exc)
