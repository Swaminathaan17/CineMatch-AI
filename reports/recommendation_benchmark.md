# Recommendation Accuracy Benchmark

- Seed movies: **11**
- K: **10**
- Label: Metadata-only proxy: shared genres/cast/director; independent of recommender score.

> ⚠️ This is an offline proxy benchmark, not human-judged relevance.

| Metric | Current | Baseline | Delta | Current 95% CI |
|---|---:|---:|---:|---|
| precision@10 | 0.9364 | 0.9182 | +0.0182 | [0.8727, 0.9818] |
| recall@10 | 0.0033 | 0.0033 | +0.0001 | [0.0023, 0.0047] |
| ndcg@10 | 0.9668 | 0.9675 | -0.0008 | [0.9272, 0.9946] |
| map@10 | 0.9481 | 0.9484 | -0.0003 | [0.8924, 0.9915] |
| mrr@10 | 0.9545 | 0.9545 | +0.0000 | [0.8636, 1.0000] |

Catalogue coverage@K: **1.24%**

## Per-seed results

- **Inception** (`27205`): NDCG 1.0000 vs baseline 1.0000
- **Interstellar** (`157336`): NDCG 0.9047 vs baseline 0.9727
- **The Dark Knight** (`155`): NDCG 0.9667 vs baseline 0.9842
- **The Martian** (`286217`): NDCG 1.0000 vs baseline 1.0000
- **Titanic** (`597`): NDCG 0.8030 vs baseline 0.7969
- **The Matrix** (`603`): NDCG 0.9895 vs baseline 0.9809
- **Avatar** (`19995`): NDCG 1.0000 vs baseline 0.9671
- **Gladiator** (`98`): NDCG 1.0000 vs baseline 1.0000
- **Toy Story** (`862`): NDCG 0.9725 vs baseline 0.9619
- **Finding Nemo** (`12`): NDCG 0.9982 vs baseline 0.9792
- **The Godfather** (`238`): NDCG 1.0000 vs baseline 1.0000
