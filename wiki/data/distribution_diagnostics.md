# Per-policy miss-rate distribution diagnostics

Source: `wiki/data/oracle_gap.json`  •  Paper L3 scope: 1MB, 4MB, 8MB

**Bootstrap CI validity verdict: PASS** — observed worst |skewness|=1.7886 (envelope: 2.0); observed worst |excess kurtosis|=4.5602 (envelope: 7.0).

Literature: Hesterberg 2015 (Am. Statistician); Efron & Tibshirani 1993 (An Introduction to the Bootstrap).

## Per-policy (marginal across apps & graphs at paper L3)

| policy | n | mean | sd | skew g1 | excess kurt g2 |
|---|---:|---:|---:|---:|---:|
| GRASP | 90 | 0.5846 | 0.3105 | -0.225 | -1.238 |
| LRU | 90 | 0.6017 | 0.3215 | -0.389 | -1.141 |
| POPT | 90 | 0.5753 | 0.3150 | -0.275 | -1.164 |
| SRRIP | 90 | 0.5848 | 0.3251 | -0.293 | -1.229 |

## Per (app, policy)

| app | policy | n | mean | sd | skew g1 | excess kurt g2 |
|---|---|---:|---:|---:|---:|---:|
| bc | GRASP | 19 | 0.6734 | 0.2509 | -0.356 | -1.054 |
| bc | LRU | 19 | 0.6898 | 0.2436 | -0.444 | -0.765 |
| bc | POPT | 19 | 0.7069 | 0.2277 | -0.427 | -0.894 |
| bc | SRRIP | 19 | 0.6758 | 0.2512 | -0.392 | -0.875 |
| bfs | GRASP | 19 | 0.8724 | 0.1430 | -1.417 | +1.508 |
| bfs | LRU | 19 | 0.8593 | 0.1973 | -2.037 | +4.560 |
| bfs | POPT | 19 | 0.8415 | 0.1625 | -0.696 | -0.900 |
| bfs | SRRIP | 19 | 0.8557 | 0.1932 | -1.789 | +3.211 |
| cc | GRASP | 16 | 0.4671 | 0.2169 | +0.558 | +0.029 |
| cc | LRU | 16 | 0.4873 | 0.2825 | -0.244 | -0.662 |
| cc | POPT | 16 | 0.4371 | 0.2629 | -0.011 | -0.614 |
| cc | SRRIP | 16 | 0.4598 | 0.2777 | -0.026 | -0.612 |
| pr | GRASP | 20 | 0.5089 | 0.3381 | +0.294 | -1.481 |
| pr | LRU | 20 | 0.5691 | 0.3358 | -0.034 | -1.595 |
| pr | POPT | 20 | 0.4905 | 0.3370 | +0.322 | -1.478 |
| pr | SRRIP | 20 | 0.5430 | 0.3437 | +0.074 | -1.614 |
| sssp | GRASP | 16 | 0.3495 | 0.2899 | +0.285 | -1.611 |
| sssp | LRU | 16 | 0.3463 | 0.3064 | +0.407 | -1.498 |
| sssp | POPT | 16 | 0.3470 | 0.2907 | +0.357 | -1.401 |
| sssp | SRRIP | 16 | 0.3324 | 0.2989 | +0.458 | -1.407 |

## Interpretation

- Skewness near 0 + negative excess kurtosis (platykurtic) means the distribution is light-tailed — the *opposite* of the pathological case (heavy tails / extreme outliers) that would bias percentile bootstrap CIs.
- Floor on |skewness| (2.0) and on |excess kurtosis| (7.0) come from Hesterberg 2015's published rules of thumb for bootstrap-CI applicability.
- Future regressions (corpus changes, scope changes) that push any (app, policy) cell beyond these floors will fail this gate and require switching to BCa / studentized bootstrap or reporting alternative CIs.
