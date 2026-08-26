# Large Recommendation Accuracy Benchmark

- Dataset size: **6856** movies
- Seed movies: **300**
- Seed strategy: reproducible stratified sample of 300 movies
- K values: **5, 10, 20**
- Label: Metadata-only proxy: shared genres/cast/director; independent of recommender score.

> ⚠️ This is an offline proxy benchmark, not human-judged relevance.

## Metrics @ 5

| Metric | Current | Baseline | Delta | Current 95% CI |
|---|---:|---:|---:|---|
| precision@5 | 0.9320 | 0.9187 | +0.0133 | [0.9153, 0.9480] |
| recall@5 | 0.0020 | 0.0019 | +0.0001 | [0.0017, 0.0024] |
| ndcg@5 | 0.9531 | 0.9389 | +0.0142 | [0.9412, 0.9644] |
| map@5 | 0.9551 | 0.9489 | +0.0062 | [0.9405, 0.9684] |
| mrr@5 | 0.9632 | 0.9601 | +0.0031 | [0.9461, 0.9783] |

## Metrics @ 10

| Metric | Current | Baseline | Delta | Current 95% CI |
|---|---:|---:|---:|---|
| precision@10 | 0.9450 | 0.9330 | +0.0120 | [0.9307, 0.9573] |
| recall@10 | 0.0041 | 0.0039 | +0.0002 | [0.0034, 0.0049] |
| ndcg@10 | 0.9535 | 0.9332 | +0.0202 | [0.9436, 0.9626] |
| map@10 | 0.9523 | 0.9431 | +0.0092 | [0.9402, 0.9635] |
| mrr@10 | 0.9632 | 0.9606 | +0.0026 | [0.9461, 0.9783] |

## Metrics @ 20

| Metric | Current | Baseline | Delta | Current 95% CI |
|---|---:|---:|---:|---|
| precision@20 | 0.9600 | 0.9505 | +0.0095 | [0.9503, 0.9688] |
| recall@20 | 0.0083 | 0.0080 | +0.0003 | [0.0069, 0.0099] |
| ndcg@20 | 0.9589 | 0.9321 | +0.0268 | [0.9511, 0.9663] |
| map@20 | 0.9540 | 0.9446 | +0.0094 | [0.9436, 0.9633] |
| mrr@20 | 0.9632 | 0.9606 | +0.0026 | [0.9461, 0.9783] |

Catalogue coverage@maxK: **53.02%**

## Largest NDCG@10 regressions/improvements

- **Scoop**: -0.3061 (0.6939 vs 1.0000)
- **Cemetery Junction**: -0.2630 (0.7370 vs 1.0000)
- **Tanguy**: -0.2206 (0.7761 vs 0.9966)
- **Elektra**: -0.1964 (0.7702 vs 0.9667)
- **The Rookie**: -0.1608 (0.8252 vs 0.9860)
- **Harley Davidson and the Marlboro Man**: -0.1485 (0.7284 vs 0.8769)
- **The Doors**: -0.1345 (0.8497 vs 0.9842)
- **Dog Pound**: -0.1231 (0.8329 vs 0.9560)
- **The Truth About Emanuel**: -0.1208 (0.7263 vs 0.8472)
- **An Inconvenient Truth**: -0.1182 (0.8352 vs 0.9534)

- **Pearl Harbor**: +0.4858 (1.0000 vs 0.5142)
- **Jakob the Liar**: +0.3770 (1.0000 vs 0.6230)
- **Weekend at Bernie's**: +0.3029 (0.8836 vs 0.5807)
- **Stories We Tell**: +0.2524 (0.7367 vs 0.4844)
- **Super High Me**: +0.2521 (1.0000 vs 0.7479)
- **The Battery**: +0.2500 (1.0000 vs 0.7500)
- **The Eye**: +0.2456 (1.0000 vs 0.7544)
- **Vacation**: +0.2449 (0.9339 vs 0.6890)
- **Rescue Dawn**: +0.2386 (0.9420 vs 0.7034)
- **Nanny McPhee and the Big Bang**: +0.2339 (1.0000 vs 0.7661)
