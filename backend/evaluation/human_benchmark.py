"""Human-rated recommendation benchmark tooling. Human labels are never fabricated."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation.benchmark import _batch_rank_baseline, _batch_rank_current, _select_stratified_seeds, _fit_metadata_baseline
from ml_engine.content_similarity import ContentSimilarityEngine

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "movies_dataset.csv"
DEFAULT_SEEDS = 40
DEFAULT_K = 10


def _bootstrap(values, n=2000, seed=42):
    if not values: return 0.0, 0.0, 0.0
    arr=np.asarray(values,float); rng=np.random.default_rng(seed)
    samples=rng.choice(arr,size=(n,len(arr)),replace=True).mean(axis=1)
    return float(arr.mean()),float(np.percentile(samples,2.5)),float(np.percentile(samples,97.5))


def _dcg(rels): return sum((2**r-1)/math.log2(i+2) for i,r in enumerate(rels))

def _ndcg(rels, all_rels):
    denom=_dcg(sorted(all_rels,reverse=True)[:len(rels)])
    return _dcg(rels)/denom if denom else 0.0

def _average_precision(rels):
    positives=sum(r>0 for r in rels)
    if not positives: return 0.0
    hits=0; total=0.0
    for i,r in enumerate(rels,1):
        if r>0:
            hits+=1; total+=hits/i
    return total/positives

def _mrr(rels):
    for i,r in enumerate(rels,1):
        if r>0: return 1.0/i
    return 0.0


def create_annotation_set(data_path=DATA_PATH, output_path=None, seed_count=DEFAULT_SEEDS, k=DEFAULT_K, random_state=42):
    df=pd.read_csv(data_path).fillna("")
    engine=ContentSimilarityEngine(df)
    _, baseline_matrix=_fit_metadata_baseline(df)
    seed_ids=_select_stratified_seeds(df,count=seed_count,random_state=random_state)
    current=_batch_rank_current(engine,seed_ids,k)
    baseline=_batch_rank_baseline(baseline_matrix,df,seed_ids,k)
    by_id={int(r.id):r for r in df.itertuples()}
    rows=[]
    for seed_id in seed_ids:
        cr={mid:rank for rank,mid in enumerate(current[seed_id],1)}
        br={mid:rank for rank,mid in enumerate(baseline[seed_id],1)}
        candidates=sorted(set(cr)|set(br),key=lambda mid:(cr.get(mid,999),br.get(mid,999)))
        seed=by_id[seed_id]
        for cid in candidates:
            cand=by_id[cid]
            rows.append({"seed_id":seed_id,"seed_title":seed.title,"candidate_id":cid,"candidate_title":cand.title,"current_rank":cr.get(cid,""),"baseline_rank":br.get(cid,""),"rating":"","annotator":"","notes":""})
    result=pd.DataFrame(rows)
    if output_path is None: output_path=Path(__file__).resolve().parents[2]/"reports"/"human_benchmark"/"human_rating_template.csv"
    output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True); result.to_csv(output_path,index=False)
    guide=output_path.parent/"HUMAN_RATING_GUIDE.md"
    guide.write_text(f'''# Human-Rated Recommendation Benchmark\n\n**{len(result):,} unique judgments** from {seed_count} seed movies. Each candidate is rated once even if both systems recommended it.\n\n## Rating scale\n- **0** — not relevant / poor recommendation\n- **1** — weakly relevant\n- **2** — good recommendation\n- **3** — excellent / highly relevant\n\nJudge only whether the candidate is a good recommendation for the seed movie. Ignore rank/system when assigning the rating.\n\n## Run\nFill the `rating` column with 0, 1, 2, or 3, then run:\n\n```bash\nPYTHONPATH=backend python backend/evaluation/human_benchmark.py evaluate --ratings reports/human_benchmark/human_rating_template.csv\n```\n\nThe evaluator reports mean relevance, Precision@10 (rating >= 2), graded NDCG@10 over the judged candidate pool, MAP@10, MRR, and paired current-vs-baseline win/tie/loss.\n\n**Ratings are intentionally blank. Do not fabricate human labels.**\n''',encoding="utf-8")
    return output_path,guide


def evaluate_annotations(ratings_path, output_dir=None, k=10):
    df=pd.read_csv(ratings_path).fillna("")
    if "rating" not in df: raise ValueError("ratings file needs a rating column")
    rated=df[df.rating.astype(str).str.strip()!=""].copy()
    if rated.empty: raise ValueError("No human ratings found. Fill rating with 0, 1, 2, or 3 first.")
    rated["rating"]=pd.to_numeric(rated.rating,errors="coerce"); rated=rated[rated.rating.between(0,3)].copy()
    if rated.empty: raise ValueError("No valid ratings. Use only 0, 1, 2, or 3.")

    def system_metrics(system):
        per_seed=[]; rank_col=system+"_rank"
        for seed_id,sg in rated.groupby("seed_id"):
            sg=sg.copy(); sg[rank_col]=pd.to_numeric(sg[rank_col],errors="coerce")
            # A seed is scored only when every candidate in the union pool has
            # been rated. This prevents partial annotation from inflating NDCG.
            total_rows_for_seed = len(df[df["seed_id"] == seed_id])
            if len(sg) < total_rows_for_seed: continue
            ranked=sg[sg[rank_col].notna()].sort_values(rank_col).head(k)
            if len(ranked)<k: continue
            rels=ranked.rating.astype(int).tolist(); all_rels=sg.rating.astype(int).tolist()
            binary=[1 if r>=2 else 0 for r in rels]
            per_seed.append({"mean_relevance":float(np.mean(rels)),"precision":sum(binary)/k,"ndcg":_ndcg(rels,all_rels),"map":_average_precision(binary),"mrr":_mrr(binary)})
        metrics={}
        for name in ("mean_relevance","precision","ndcg","map","mrr"):
            vals=[x[name] for x in per_seed]; mean,lo,hi=_bootstrap(vals); metrics[name]={"mean":round(mean,4),"ci95":[round(lo,4),round(hi,4)]}
        return {"complete_seed_count":len(per_seed),"metrics":metrics}

    systems={"current":system_metrics("current"),"baseline":system_metrics("baseline")}
    paired_rows=[]
    for seed_id,sg in rated.groupby("seed_id"):
        c=sg[pd.to_numeric(sg.current_rank,errors="coerce").notna()].sort_values("current_rank").head(k)
        b=sg[pd.to_numeric(sg.baseline_rank,errors="coerce").notna()].sort_values("baseline_rank").head(k)
        if len(c)==k and len(b)==k: paired_rows.append({"seed_id":seed_id,"current":c.rating.mean(),"baseline":b.rating.mean()})
    paired=pd.DataFrame(paired_rows); pairwise={}
    if not paired.empty:
        pairwise={"paired_seed_count":int(len(paired)),"current_win_rate":round(float((paired.current>paired.baseline).mean()),4),"tie_rate":round(float((paired.current==paired.baseline).mean()),4),"baseline_win_rate":round(float((paired.baseline>paired.current).mean()),4),"mean_relevance_current":round(float(paired.current.mean()),4),"mean_relevance_baseline":round(float(paired.baseline.mean()),4)}
    agreement={}
    if "annotator" in rated.columns and rated["annotator"].astype(str).str.strip().nunique() >= 2:
        a=rated.copy(); a["annotator"]=a["annotator"].astype(str).str.strip(); a=a[a["annotator"]!=""]
        names=sorted(a["annotator"].unique())[:2]
        if len(names)==2:
            left=a[a.annotator==names[0]][["seed_id","candidate_id","rating"]].rename(columns={"rating":"r1"})
            right=a[a.annotator==names[1]][["seed_id","candidate_id","rating"]].rename(columns={"rating":"r2"})
            merged=left.merge(right,on=["seed_id","candidate_id"],how="inner")
            if len(merged)>1:
                observed=float((merged.r1==merged.r2).mean())
                p1=merged.r1.value_counts(normalize=True); p2=merged.r2.value_counts(normalize=True)
                expected=sum(float(p1.get(v,0))*float(p2.get(v,0)) for v in range(4))
                kappa=(observed-expected)/(1-expected) if expected<1 else 1.0
                agreement={"annotators":names,"paired_items":int(len(merged)),"percent_agreement":round(observed,4),"cohen_kappa":round(float(kappa),4)}
    result={"benchmark":{"ratings_file":str(ratings_path),"rated_rows":int(len(rated)),"total_rows":int(len(df)),"seed_count":int(rated.seed_id.nunique()),"scale":"0=not relevant, 1=weak, 2=good, 3=excellent","precision_threshold":"rating >= 2","ndcg_pool":"all human-rated candidates for each seed"},"systems":systems,"pairwise":pairwise,"inter_rater_agreement":agreement,"warning":"Human quality depends on annotator consistency and sample selection. Complete all candidate ratings for each seed before reporting final metrics."}
    out=Path(output_dir) if output_dir else Path(ratings_path).resolve().parent; out.mkdir(parents=True,exist_ok=True)
    (out/"human_benchmark_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    lines=["# Human-Rated Benchmark Results","",f"Rated rows: **{len(rated):,} / {len(df):,}**",""]
    for system,data in systems.items():
        lines += [f"## {system}","","| Metric | Mean | 95% CI |","|---|---:|---|"]
        for metric,vals in data["metrics"].items(): lines.append(f"| {metric}@{k} | {vals['mean']:.4f} | [{vals['ci95'][0]:.4f}, {vals['ci95'][1]:.4f}] |")
        lines.append("")
    if pairwise: lines += ["## Paired human preference","",f"- Current wins: **{pairwise['current_win_rate']:.2%}**",f"- Ties: **{pairwise['tie_rate']:.2%}**",f"- Baseline wins: **{pairwise['baseline_win_rate']:.2%}**",""]
    (out/"human_benchmark_results.md").write_text("\n".join(lines),encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    create=sub.add_parser("create"); create.add_argument("--data",default=str(DATA_PATH)); create.add_argument("--output",default=str(Path(__file__).resolve().parents[2]/"reports"/"human_benchmark"/"human_rating_template.csv")); create.add_argument("--seed-count",type=int,default=DEFAULT_SEEDS); create.add_argument("--k",type=int,default=DEFAULT_K); create.add_argument("--random-state",type=int,default=42)
    evaluate=sub.add_parser("evaluate"); evaluate.add_argument("--ratings",required=True); evaluate.add_argument("--output-dir",default=None); evaluate.add_argument("--k",type=int,default=10)
    args=parser.parse_args()
    if args.command=="create":
        paths=create_annotation_set(args.data,args.output,args.seed_count,args.k,args.random_state); print(paths[0]); print(paths[1])
    else: print(json.dumps(evaluate_annotations(args.ratings,args.output_dir,args.k),indent=2))

if __name__=="__main__": main()
