# Large Recommendation Accuracy Benchmark

- Dataset size: **6856** movies
- Seed movies: **500**
- Seed strategy: reproducible stratified sample of 500 movies
- K values: **5, 10, 20**
- Label: Metadata-only proxy: shared genres/cast/director; independent of recommender score.

> ⚠️ This is an offline proxy benchmark, not human-judged relevance.

## Metrics @ 5

| Metric | Current | Baseline | Delta | Current 95% CI |
|---|---:|---:|---:|---|
| precision@5 | 0.9420 | 0.9264 | +0.0156 | [0.9300, 0.9548] |
| recall@5 | 0.0019 | 0.0018 | +0.0001 | [0.0016, 0.0021] |
| ndcg@5 | 0.9666 | 0.9521 | +0.0145 | [0.9584, 0.9746] |
| map@5 | 0.9678 | 0.9607 | +0.0072 | [0.9581, 0.9770] |
| mrr@5 | 0.9744 | 0.9723 | +0.0021 | [0.9636, 0.9847] |

## Metrics @ 10

| Metric | Current | Baseline | Delta | Current 95% CI |
|---|---:|---:|---:|---|
| precision@10 | 0.9478 | 0.9318 | +0.0160 | [0.9382, 0.9572] |
| recall@10 | 0.0038 | 0.0036 | +0.0002 | [0.0033, 0.0043] |
| ndcg@10 | 0.9651 | 0.9408 | +0.0244 | [0.9587, 0.9714] |
| map@10 | 0.9604 | 0.9503 | +0.0101 | [0.9521, 0.9688] |
| mrr@10 | 0.9747 | 0.9728 | +0.0019 | [0.9640, 0.9849] |

## Metrics @ 20

| Metric | Current | Baseline | Delta | Current 95% CI |
|---|---:|---:|---:|---|
| precision@20 | 0.9620 | 0.9495 | +0.0125 | [0.9547, 0.9689] |
| recall@20 | 0.0077 | 0.0075 | +0.0002 | [0.0068, 0.0087] |
| ndcg@20 | 0.9660 | 0.9365 | +0.0296 | [0.9609, 0.9712] |
| map@20 | 0.9592 | 0.9479 | +0.0113 | [0.9522, 0.9664] |
| mrr@20 | 0.9747 | 0.9728 | +0.0019 | [0.9640, 0.9849] |

Catalogue coverage@maxK: **68.04%**

## Largest NDCG@10 regressions/improvements

- **Blue Crush**: -0.2688 (0.6883 vs 0.9571)
- **Cemetery Junction**: -0.2630 (0.7370 vs 1.0000)
- **Corpse Bride**: -0.1849 (0.8151 vs 1.0000)
- **Crocodile Dundee in Los Angeles**: -0.1686 (0.7764 vs 0.9450)
- **The Rookie**: -0.1608 (0.8252 vs 0.9860)
- **The Cook, the Thief, His Wife & Her Lover**: -0.1225 (0.8414 vs 0.9639)
- **Le Silence de Lorna**: -0.1148 (0.8039 vs 0.9187)
- **Cape Fear**: -0.1056 (0.5060 vs 0.6116)
- **The Messengers**: -0.1029 (0.7284 vs 0.8313)
- **Action Jackson**: -0.1026 (0.8071 vs 0.9098)

- **The Wiz**: +0.6629 (0.9520 vs 0.2891)
- **Angry Video Game Nerd: The Movie**: +0.4678 (0.9872 vs 0.5194)
- **Poupoupidou**: +0.4126 (0.9976 vs 0.5850)
- **The Overnighters**: +0.3468 (1.0000 vs 0.6532)
- **Cherrybomb**: +0.3019 (1.0000 vs 0.6981)
- **Liberty Stands Still**: +0.2693 (0.9613 vs 0.6920)
- **Death Race 2000**: +0.2642 (1.0000 vs 0.7358)
- **Obvious Child**: +0.2460 (0.9677 vs 0.7217)
- **The Boy Next Door**: +0.2448 (0.9953 vs 0.7505)
- **Outsourced**: +0.2373 (1.0000 vs 0.7627)
