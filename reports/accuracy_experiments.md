# Accuracy Experiments Log

## Baseline

**Date:** 2026-08-26
**Dataset:** 6856 movies, 500 stratified seeds
**Labels:** Metadata-only proxy (genres/cast/director)

### Accuracy
| Metric | @5 | @10 | @20 |
|---|---|---|---|
| Precision | 0.9420 | 0.9478 | 0.9620 |
| NDCG | 0.9666 | 0.9651 | 0.9660 |
| MAP | 0.9678 | 0.9604 | 0.9592 |
| MRR | 0.9744 | 0.9747 | 0.9747 |

### Speed
- Engine build: 1.85s
- Single rec latency (top-10): 20.9ms
- Benchmark runtime: 25.5s
- Meta vocab: 98,271 | Overview vocab: 184,877

---

## Experiment 1: Fix double-counted preference

**Proposal:** #1
**Files changed:** `backend/app/services/recommendation_service.py` (line 162)
**What changed:** `content_similarity=pref` → `content_similarity=0.0` in `rank_personalized_candidates`. The same `pref` value was being passed as both `content_similarity` (weight 0.42) and `preference_match` (weight 0.24), giving it 75% effective influence. Now preference_match carries the signal alone (~30% after redistribution).

**Result:** KEEP — correctness fix, no benchmark regression (benchmark tests content retrieval only, not personalized path). 31/31 tests pass.

---

## Experiment 2: Raise min_df to 2

**Proposal:** #2
**Files changed:** `backend/ml_engine/content_similarity.py`
**What changed:** min_df=1 → min_df=2 on both vectorizers, filtering singleton terms.

**Results:**
| Metric | @10 Before | @10 After | Delta |
|---|---|---|---|
| Precision | 0.9478 | 0.9516 | +0.004 |
| NDCG | 0.9651 | 0.9613 | -0.004 |
| MAP | 0.9604 | 0.9597 | -0.001 |
| Coverage | 0.6804 | 0.6246 | -0.056 |

**Result:** REVERT — NDCG regression, coverage dropped 5.6pp.

---

## Experiment 3: Increase overview contribution to 40%

**Proposal:** #3
**Files changed:** `backend/ml_engine/content_similarity.py`
**What changed:** _combine from 0.72/0.28 → 0.60/0.40 meta/overview blend.

**Results:**
| Metric | @10 Before | @10 After | Delta |
|---|---|---|---|
| Precision | 0.9478 | 0.9460 | -0.002 |
| NDCG | 0.9651 | 0.9630 | -0.002 |
| MAP | 0.9604 | 0.9597 | -0.001 |
| Coverage | 0.6804 | 0.6941 | +0.014 |

**Result:** REVERT — marginal NDCG regression, no accuracy improvement.

---

## Experiment 4: Preprocess overviews

**Proposal:** #4
**Files changed:** `backend/ml_engine/content_similarity.py`
**What changed:** Added HTML entity removal and whitespace normalization to `_overview()`. Added `import re` and `_HTML_ENTITY` regex.

**Results:** Identical metrics across all K values. No regression, no improvement.

**Result:** KEEP — correctness improvement (removes HTML entities, normalizes whitespace), zero cost.

---

## Experiment 5: Combined profile vector

**Proposal:** #5
**Skipped:** Personalization path only — not measured by the benchmark (which tests content retrieval).

---

## Experiment 6: Add release date

**Proposal:** #6
**Skipped:** Dataset (`movies_dataset.csv`) has no `release_date` column. Would require dataset modification.

---

## Experiment 7: Cap overview max_features

**Proposal:** #7
**Files changed:** `backend/ml_engine/content_similarity.py`
**What changed:** Added `max_features=5000` to overview_vectorizer.

**Results:**
| Metric | @10 Before | @10 After | Delta |
|---|---|---|---|
| Precision | 0.9484 | 0.9464 | -0.002 |
| NDCG | 0.9650 | 0.9563 | -0.009 |
| MAP | 0.9605 | 0.9595 | -0.001 |
| Coverage | 0.6804 | 0.7144 | +0.034 |
| Runtime | ~30s | 76.1s | +153% |

**Result:** REVERT — NDCG regression -0.009, runtime tripled. Sparse matrix becomes less efficient with truncated vocabulary.

---

## Experiment 8: Improve benchmark relevance labels

**Proposal:** #8
**Files changed:** `backend/evaluation/benchmark.py`
**What changed:** Added overview text cosine similarity to relevance labels. Movies with high overview similarity (>=0.35 → rel=2, >=0.20 → rel=1) get relevance even without metadata overlap. Precomputed overview TF-IDF matrix (5000 features) in `_build_relevance_cache`.

**Results:** Near-identical metrics (all within CI). The overview labels add a small amount of signal where metadata was 0.

**Result:** KEEP — more complete relevance labels for future experiments. New baseline established.

### New Baseline (post-Proposals 1, 4, 8, 9)
| Metric | @5 | @10 | @20 |
|---|---|---|---|
| Precision | 0.9424 | 0.9484 | 0.9624 |
| NDCG | 0.9666 | 0.9650 | 0.9659 |
| MAP | 0.9678 | 0.9605 | 0.9595 |
| MRR | 0.9744 | 0.9747 | 0.9747 |
| Coverage | — | — | 0.6804 |

---

## Experiment 9: Align NL query vectorizer

**Proposal:** #9
**Files changed:** `backend/ml_engine/nl_query_parser.py`
**What changed:** NL query fallback vectorizer aligned with content engine: `max_features=3000, ngram_range=(1,1)` → `max_features=5000, ngram_range=(1,2), sublinear_tf=True`.

**Result:** KEEP — no benchmark impact (NL search path), but improves NL query quality. 31/31 tests pass.

---

## Experiment 10: Reduce negative subtraction

**Proposal:** #10
**Skipped:** Personalization path only — not measured by the benchmark.

---

## Final Summary

### Changes Kept
1. **Proposal 1:** Fixed double-counted preference in `recommendation_service.py`
2. **Proposal 4:** Conservative overview preprocessing (HTML entities, whitespace)
3. **Proposal 8:** Improved benchmark relevance labels (metadata + overview similarity)
4. **Proposal 9:** Aligned NL query vectorizer with content engine

### Changes Reverted
- Proposal 2 (min_df=2): -0.004 NDCG, -5.6pp coverage
- Proposal 3 (overview 40%): -0.002 NDCG
- Proposal 7 (overview max_features=5000): -0.009 NDCG, 3x slower

### Changes Skipped
- Proposal 5, 10: Personalization path (not benchmark-measured)
- Proposal 6: No release_date in dataset

### Final Metrics (500 seeds, 6856 movies)
| Metric | @5 | @10 | @20 |
|---|---|---|---|
| Precision | 0.9424 | 0.9484 | 0.9624 |
| NDCG | 0.9666 | 0.9650 | 0.9659 |
| MAP | 0.9678 | 0.9605 | 0.9595 |
| MRR | 0.9744 | 0.9747 | 0.9747 |
| Coverage@20 | — | — | 0.6804 |

### Final Speed
- Engine build: 2.01s
- Single rec latency: 22.3ms
- Batch similarity: 0.02s
- Benchmark runtime: ~35s
