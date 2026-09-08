"""Offline benchmarking for the movie recommender.

This benchmark intentionally does not use the recommender's own scores to
create labels. Relevance is derived only from independent catalogue metadata
(genre/director/cast overlap), so the benchmark can detect ranking regressions.
It is a proxy benchmark, not a claim of human-judged relevance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ml_engine.content_similarity import ContentSimilarityEngine, _clean_list_field, build_soup

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "movies_dataset.csv"
DEFAULT_SEED_COUNT = 500
DEFAULT_KS = [5, 10, 20]


def _tokens(value) -> set[str]:
    return {x.strip().lower() for x in _clean_list_field(value) if x.strip()}


def _gold_relevance(seed: pd.Series, candidate: pd.Series) -> int:
    """Independent relevance proxy: 0..3, based only on metadata overlap."""
    if int(seed["id"]) == int(candidate["id"]):
        return 0
    seed_genres, cand_genres = _tokens(seed.get("genres")), _tokens(candidate.get("genres"))
    shared_genres = len(seed_genres & cand_genres)
    same_director = bool(_tokens(seed.get("director")) & _tokens(candidate.get("director")))
    shared_cast = len(_tokens(seed.get("cast")) & _tokens(candidate.get("cast")))

    # Tier 3: strong structural relationship.
    if same_director and shared_genres >= 1:
        return 3
    if shared_genres >= 3:
        return 3
    # Tier 2: meaningful multi-attribute overlap.
    if shared_genres >= 2 or shared_cast >= 2:
        return 2
    # Tier 1: weak but valid topical relation.
    if shared_genres >= 1 or shared_cast >= 1:
        return 1
    return 0


def precision_at_k(rels: list[int], k: int) -> float:
    top = rels[:k]
    return sum(r > 0 for r in top) / k if k else 0.0


def recall_at_k(rels: list[int], total_relevant: int, k: int) -> float:
    return sum(r > 0 for r in rels[:k]) / total_relevant if total_relevant else 0.0


def ndcg_at_k(rels: list[int], k: int) -> float:
    gains = [2**r - 1 for r in rels[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(rels, reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def average_precision_at_k(rels: list[int], k: int) -> float:
    total = sum(r > 0 for r in rels)
    if not total:
        return 0.0
    hits = 0
    score = 0.0
    for i, r in enumerate(rels[:k], start=1):
        if r > 0:
            hits += 1
            score += hits / i
    return score / min(total, k)


def mrr(rels: list[int]) -> float:
    for i, r in enumerate(rels, start=1):
        if r > 0:
            return 1.0 / i
    return 0.0


def _fit_metadata_baseline(df: pd.DataFrame):
    """Fit the legacy metadata-only baseline once for a fair large benchmark."""
    soup = df.apply(build_soup, axis=1)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
    mat = vec.fit_transform(soup)
    return vec, mat


def _rank_from_matrix(matrix, idx: int, top_n: int) -> list[int]:
    scores = cosine_similarity(matrix[idx], matrix)[0]
    order = np.argsort(-scores)
    return [int(i) for i in order if int(i) != idx][:top_n]


def _metadata_baseline(matrix, seed_idx: int, top_n: int) -> list[int]:
    """Legacy-style TF-IDF baseline: structured metadata only."""
    return _rank_from_matrix(matrix, seed_idx, top_n)


def _select_stratified_seeds(df: pd.DataFrame, count: int = DEFAULT_SEED_COUNT, random_state: int = 42) -> list[int]:
    """Select a reproducible, broad seed set instead of cherry-picking famous movies.

    The sample is balanced across rating and popularity quartiles, then supplemented
    to cover as many genres as possible. This makes the large benchmark more useful
    for regression detection than a hand-picked list of blockbuster seeds.
    """
    work = df.copy()
    work["_rating_bin"] = pd.qcut(work["vote_average"].rank(method="first"), 4, labels=False)
    work["_pop_bin"] = pd.qcut(work["popularity"].rank(method="first"), 4, labels=False)
    rng = np.random.default_rng(random_state)
    target = min(int(count), len(work))
    selected: list[int] = []
    # 16 strata: rating x popularity. Allocate approximately evenly.
    groups = list(work.groupby(["_rating_bin", "_pop_bin"], sort=True, observed=True))
    per_group = max(1, target // max(1, len(groups)))
    for _, g in groups:
        take = min(per_group, len(g))
        idx = rng.choice(g.index.to_numpy(), size=take, replace=False)
        selected.extend(work.loc[idx, "id"].astype(int).tolist())
    # Fill remainder from a random catalogue sample.
    remaining = [int(x) for x in work["id"] if int(x) not in set(selected)]
    if len(selected) < target:
        extra = rng.choice(np.asarray(remaining), size=target - len(selected), replace=False)
        selected.extend(int(x) for x in extra)
    return selected[:target]


def _build_relevance_cache(df: pd.DataFrame):
    """Precompute compact token sets so large benchmarks avoid O(n^2) parsing overhead."""
    overview_vec = TfidfVectorizer(stop_words="english", max_features=5000)
    overview_texts = df.get("overview", pd.Series([""] * len(df))).fillna("").str.strip().str.lower()
    overview_mat = overview_vec.fit_transform(overview_texts)

    cache = {}
    for idx, row in df.iterrows():
        cache[int(row["id"])] = {
            "idx": idx,
            "genres": _tokens(row.get("genres")),
            "director": _tokens(row.get("director")),
            "cast": _tokens(row.get("cast")),
        }
    return cache, overview_mat


def _gold_relevance_cached(seed_id: int, candidate_id: int, cache: dict, overview_sim: float | None = None) -> int:
    if seed_id == candidate_id:
        return 0
    seed, cand = cache[seed_id], cache[candidate_id]
    shared_genres = len(seed["genres"] & cand["genres"])
    same_director = bool(seed["director"] & cand["director"])
    shared_cast = len(seed["cast"] & cand["cast"])

    meta_score = 0
    if same_director and shared_genres >= 1:
        meta_score = 3
    elif shared_genres >= 3:
        meta_score = 3
    elif shared_genres >= 2 or shared_cast >= 2:
        meta_score = 2
    elif shared_genres >= 1 or shared_cast >= 1:
        meta_score = 1

    if meta_score > 0:
        return meta_score

    if overview_sim is not None:
        if overview_sim >= 0.35:
            return 2
        if overview_sim >= 0.20:
            return 1

    return 0


def _metric_row(recommendations: list[int], seed_id: int, df: pd.DataFrame, cache: dict, k: int, overview_matrix=None, id_to_idx=None) -> dict:
    if overview_matrix is not None and id_to_idx is not None and seed_id in id_to_idx:
        seed_idx = id_to_idx[seed_id]
        cand_idxs = [id_to_idx[rid] for rid in recommendations if rid in id_to_idx]
        if cand_idxs:
            sims = cosine_similarity(overview_matrix[seed_idx], overview_matrix[cand_idxs])[0]
            sim_map = {recommendations[i]: float(sims[i]) for i in range(min(len(cand_idxs), len(recommendations)))}
        else:
            sim_map = {}
    else:
        sim_map = {}
    relevant = [_gold_relevance_cached(seed_id, rid, cache, sim_map.get(rid)) for rid in recommendations]
    total_relevant = sum(
        _gold_relevance_cached(seed_id, int(candidate_id), cache) > 0
        for candidate_id in cache
        if candidate_id != seed_id
    )
    return {
        "precision": precision_at_k(relevant, k),
        "recall": recall_at_k(relevant, total_relevant, k),
        "ndcg": ndcg_at_k(relevant, k),
        "map": average_precision_at_k(relevant, k),
        "mrr": mrr(relevant),
    }


def _bootstrap_mean(values: list[float], n: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    samples = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))



def _batch_rank_current(engine: ContentSimilarityEngine, seed_ids: list[int], top_n: int, batch_size: int = 64) -> dict[int, list[int]]:
    """Rank many seeds in batches to make 500+/1000-movie benchmarks practical."""
    id_to_idx = {int(mid): i for i, mid in enumerate(engine.movies_df["id"].astype(int))}
    result = {}
    for start in range(0, len(seed_ids), batch_size):
        batch_ids = seed_ids[start:start + batch_size]
        idxs = [id_to_idx[mid] for mid in batch_ids]
        meta_scores = cosine_similarity(engine.meta_matrix[idxs], engine.meta_matrix)
        if engine._has_overview_vocab:
            overview_scores = cosine_similarity(engine.overview_matrix[idxs], engine.overview_matrix)
        else:
            overview_scores = np.zeros_like(meta_scores)
        scores = engine._combine(meta_scores, overview_scores)
        for row_pos, seed_id in enumerate(batch_ids):
            seed_idx = idxs[row_pos]
            row_scores = scores[row_pos].copy()
            row_scores[seed_idx] = -np.inf
            # argpartition avoids a full O(N log N) sort for every seed.
            candidate_n = min(top_n, len(row_scores) - 1)
            part = np.argpartition(-row_scores, candidate_n - 1)[:candidate_n]
            ordered = part[np.argsort(-row_scores[part])]
            result[seed_id] = [int(engine.movies_df.iloc[i]["id"]) for i in ordered]
    return result


def _batch_rank_baseline(matrix, df: pd.DataFrame, seed_ids: list[int], top_n: int, batch_size: int = 64) -> dict[int, list[int]]:
    id_to_idx = {int(mid): i for i, mid in enumerate(df["id"].astype(int))}
    result = {}
    for start in range(0, len(seed_ids), batch_size):
        batch_ids = seed_ids[start:start + batch_size]
        idxs = [id_to_idx[mid] for mid in batch_ids]
        scores = cosine_similarity(matrix[idxs], matrix)
        for row_pos, seed_id in enumerate(batch_ids):
            seed_idx = idxs[row_pos]
            row_scores = scores[row_pos].copy()
            row_scores[seed_idx] = -np.inf
            candidate_n = min(top_n, len(row_scores) - 1)
            part = np.argpartition(-row_scores, candidate_n - 1)[:candidate_n]
            ordered = part[np.argsort(-row_scores[part])]
            result[seed_id] = [int(df.iloc[i]["id"]) for i in ordered]
    return result

def run_benchmark(
    data_path: str | Path = DATA_PATH,
    seeds: Iterable[int] | None = None,
    k: int = 10,
    seed_count: int = DEFAULT_SEED_COUNT,
    ks: Iterable[int] | None = None,
) -> dict:
    df = pd.read_csv(data_path).fillna("")
    ids = set(df["id"].astype(int))
    if seeds is None:
        seed_ids = _select_stratified_seeds(df, seed_count)
        seed_source = f"reproducible stratified sample of {len(seed_ids)} movies"
    else:
        seed_ids = [int(x) for x in seeds if int(x) in ids]
        seed_source = "explicit seed IDs"
    if len(seed_ids) < 5:
        raise ValueError("Benchmark needs at least 5 seed movies present in the dataset")

    eval_ks = sorted(set(int(x) for x in (ks if ks is not None else [k])))
    max_k = max(eval_ks)
    engine = ContentSimilarityEngine(df)
    _, baseline_matrix = _fit_metadata_baseline(df)
    cache, overview_matrix = _build_relevance_cache(df)
    by_id_idx = {int(movie_id): idx for idx, movie_id in enumerate(df["id"].astype(int))}
    current_rankings = _batch_rank_current(engine, seed_ids, max_k)
    baseline_rankings = _batch_rank_baseline(baseline_matrix, df, seed_ids, max_k)
    rows = []
    for seed_id in seed_ids:
        seed = df[df["id"] == seed_id].iloc[0]
        current_ids = current_rankings[seed_id]
        baseline_ids = baseline_rankings[seed_id]
        per_k = {}
        for current_k in eval_ks:
            per_k[str(current_k)] = {
                "current": _metric_row(current_ids[:current_k], seed_id, df, cache, current_k, overview_matrix, by_id_idx),
                "baseline": _metric_row(baseline_ids[:current_k], seed_id, df, cache, current_k, overview_matrix, by_id_idx),
            }
        rows.append({"seed_id": seed_id, "title": seed["title"], "metrics": per_k})

    metrics = {}
    for current_k in eval_ks:
        metrics[str(current_k)] = {}
        for metric in ["precision", "recall", "ndcg", "map", "mrr"]:
            cur_values = [r["metrics"][str(current_k)]["current"][metric] for r in rows]
            old_values = [r["metrics"][str(current_k)]["baseline"][metric] for r in rows]
            cmean, clo, chi = _bootstrap_mean(cur_values)
            bmean, blo, bhi = _bootstrap_mean(old_values)
            metrics[str(current_k)][metric] = {
                "current": round(cmean, 4),
                "current_ci95": [round(clo, 4), round(chi, 4)],
                "baseline": round(bmean, 4),
                "baseline_ci95": [round(blo, 4), round(bhi, 4)],
                "delta": round(cmean - bmean, 4),
            }

    current_ids = []
    for seed_id in seed_ids:
        current_ids.extend(current_rankings[seed_id])
    coverage = len(set(current_ids)) / max(1, len(df))
    public_metrics = metrics[str(eval_ks[0])] if len(eval_ks) == 1 and ks is None else metrics
    return {
        "benchmark": {
            "dataset": str(data_path),
            "dataset_size": int(len(df)),
            "seed_count": len(seed_ids),
            "seed_source": seed_source,
            "k_values": eval_ks,
            "label_definition": "Metadata-only proxy: shared genres/cast/director; independent of recommender score.",
            "warning": "This is an offline proxy benchmark, not human-judged relevance. Use human evaluation before claiming production accuracy.",
        },
        "metrics": public_metrics,
        "diagnostics": {
            "catalog_coverage_at_max_k": round(coverage, 6),
            "unique_recommended_movies_at_max_k": len(set(current_ids)),
        },
        "cases": rows,
    }

def write_report(result: dict, output_dir: str | Path) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "recommendation_benchmark.json"
    md_path = out / "recommendation_benchmark.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Large Recommendation Accuracy Benchmark",
        "",
        f"- Dataset size: **{result['benchmark']['dataset_size']}** movies",
        f"- Seed movies: **{result['benchmark']['seed_count']}**",
        f"- Seed strategy: {result['benchmark']['seed_source']}",
        f"- K values: **{', '.join(map(str, result['benchmark']['k_values']))}**",
        f"- Label: {result['benchmark']['label_definition']}",
        "",
        "> ⚠️ This is an offline proxy benchmark, not human-judged relevance.",
        "",
    ]
    for k in result["benchmark"]["k_values"]:
        lines += [f"## Metrics @ {k}", "", "| Metric | Current | Baseline | Delta | Current 95% CI |", "|---|---:|---:|---:|---|"]
        for name, m in result["metrics"][str(k)].items():
            lines.append(f"| {name}@{k} | {m['current']:.4f} | {m['baseline']:.4f} | {m['delta']:+.4f} | [{m['current_ci95'][0]:.4f}, {m['current_ci95'][1]:.4f}] |")
        lines.append("")
    lines += [f"Catalogue coverage@maxK: **{result['diagnostics']['catalog_coverage_at_max_k']:.2%}**", "", "## Largest NDCG@10 regressions/improvements", ""]
    k_for_cases = "10" if "10" in result["metrics"] else str(result["benchmark"]["k_values"][-1])
    case_deltas = []
    for case in result["cases"]:
        c = case["metrics"][k_for_cases]["current"]["ndcg"]
        b = case["metrics"][k_for_cases]["baseline"]["ndcg"]
        case_deltas.append((c - b, case["title"], c, b))
    for delta, title, c, b in sorted(case_deltas)[:10]:
        lines.append(f"- **{title}**: {delta:+.4f} ({c:.4f} vs {b:.4f})")
    lines.append("")
    for delta, title, c, b in sorted(case_deltas, reverse=True)[:10]:
        lines.append(f"- **{title}**: {delta:+.4f} ({c:.4f} vs {b:.4f})")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[2] / "reports"))
    args = parser.parse_args()
    result = run_benchmark(args.data, k=args.k, seed_count=args.seed_count, ks=args.ks)
    paths = write_report(result, args.output)
    print(json.dumps(result["metrics"], indent=2))
    print(f"Wrote {paths[0]}")
    print(f"Wrote {paths[1]}")


if __name__ == "__main__":
    main()

