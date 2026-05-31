# Per-policy miss-rate distribution diagnostics

Source: `wiki/data/oracle_gap.json`  •  Paper L3 scope: 1MB, 4MB, 8MB

**Bootstrap CI validity verdict: PASS** — observed worst |skewness|=1.1502 (envelope: 2.0); observed worst |excess kurtosis|=1.6021 (envelope: 7.0).

Literature: Hesterberg 2015 (Am. Statistician); Efron & Tibshirani 1993 (An Introduction to the Bootstrap).

## Per-policy (marginal across apps & graphs at paper L3)

| policy | n | mean | sd | skew g1 | excess kurt g2 |
|---|---:|---:|---:|---:|---:|
| GRASP | 90 | 0.4826 | 0.3235 | +0.019 | -1.340 |
| LRU | 90 | 0.5100 | 0.3249 | -0.112 | -1.290 |
| POPT | 90 | 0.4727 | 0.3166 | +0.009 | -1.297 |
| SRRIP | 90 | 0.4930 | 0.3245 | -0.020 | -1.297 |

## Per (app, policy)

| app | policy | n | mean | sd | skew g1 | excess kurt g2 |
|---|---|---:|---:|---:|---:|---:|
| bc | GRASP | 19 | 0.4759 | 0.3071 | -0.248 | -1.176 |
| bc | LRU | 19 | 0.4897 | 0.2992 | -0.413 | -0.947 |
| bc | POPT | 19 | 0.4815 | 0.2973 | -0.414 | -1.090 |
| bc | SRRIP | 19 | 0.4726 | 0.2942 | -0.343 | -1.012 |
| bfs | GRASP | 19 | 0.6913 | 0.3051 | -1.150 | +0.323 |
| bfs | LRU | 19 | 0.6915 | 0.3450 | -1.093 | -0.249 |
| bfs | POPT | 19 | 0.6649 | 0.3171 | -1.090 | +0.187 |
| bfs | SRRIP | 19 | 0.6883 | 0.3412 | -1.073 | -0.230 |
| cc | GRASP | 16 | 0.3674 | 0.2483 | +0.304 | -0.871 |
| cc | LRU | 16 | 0.4419 | 0.2517 | -0.433 | -1.034 |
| cc | POPT | 16 | 0.4055 | 0.2581 | -0.019 | -1.310 |
| cc | SRRIP | 16 | 0.4154 | 0.2475 | -0.231 | -1.177 |
| pr | GRASP | 20 | 0.4968 | 0.3436 | +0.264 | -1.499 |
| pr | LRU | 20 | 0.5566 | 0.3377 | +0.012 | -1.602 |
| pr | POPT | 20 | 0.4694 | 0.3408 | +0.399 | -1.409 |
| pr | SRRIP | 20 | 0.5312 | 0.3445 | +0.118 | -1.597 |
| sssp | GRASP | 16 | 0.3403 | 0.3060 | +0.431 | -1.310 |
| sssp | LRU | 16 | 0.3284 | 0.2882 | +0.432 | -1.352 |
| sssp | POPT | 16 | 0.3055 | 0.2695 | +0.477 | -1.152 |
| sssp | SRRIP | 16 | 0.3151 | 0.2807 | +0.482 | -1.231 |

## Interpretation

- Skewness near 0 + negative excess kurtosis (platykurtic) means the distribution is light-tailed — the *opposite* of the pathological case (heavy tails / extreme outliers) that would bias percentile bootstrap CIs.
- Floor on |skewness| (2.0) and on |excess kurtosis| (7.0) come from Hesterberg 2015's published rules of thumb for bootstrap-CI applicability.
- Future regressions (corpus changes, scope changes) that push any (app, policy) cell beyond these floors will fail this gate and require switching to BCa / studentized bootstrap or reporting alternative CIs.
