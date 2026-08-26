# Human-Rated Recommendation Benchmark

**568 unique judgments** from 40 seed movies. Each candidate is rated once even if both systems recommended it.

## Rating scale
- **0** — not relevant / poor recommendation
- **1** — weakly relevant
- **2** — good recommendation
- **3** — excellent / highly relevant

Judge only whether the candidate is a good recommendation for the seed movie. Ignore rank/system when assigning the rating.

## Run
Fill the `rating` column with 0, 1, 2, or 3, then run:

```bash
PYTHONPATH=backend python backend/evaluation/human_benchmark.py evaluate --ratings reports/human_benchmark/human_rating_template.csv
```

The evaluator reports mean relevance, Precision@10 (rating >= 2), graded NDCG@10 over the judged candidate pool, MAP@10, MRR, and paired current-vs-baseline win/tie/loss.

**Ratings are intentionally blank. Do not fabricate human labels.**
