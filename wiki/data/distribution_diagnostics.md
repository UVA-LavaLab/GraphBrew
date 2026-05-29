# Per-policy miss-rate distribution diagnostics

Source: `wiki/data/oracle_gap.json`  •  Paper L3 scope: 1MB, 4MB, 8MB

**Bootstrap CI validity verdict: PASS** — observed worst |skewness|=1.2824 (envelope: 2.0); observed worst |excess kurtosis|=1.3631 (envelope: 7.0).

Literature: Hesterberg 2015 (Am. Statistician); Efron & Tibshirani 1993 (An Introduction to the Bootstrap).

## Per-policy (marginal across apps & graphs at paper L3)

| policy | n | mean | sd | skew g1 | excess kurt g2 |
|---|---:|---:|---:|---:|---:|
| GRASP | 90 | 0.4525 | 0.3202 | +0.056 | -1.378 |
| LRU | 90 | 0.4775 | 0.3234 | -0.059 | -1.339 |
| POPT | 90 | 0.4422 | 0.3120 | +0.034 | -1.357 |
| SRRIP | 90 | 0.4614 | 0.3216 | +0.026 | -1.330 |

## Per (app, policy)

| app | policy | n | mean | sd | skew g1 | excess kurt g2 |
|---|---|---:|---:|---:|---:|---:|
| bc | GRASP | 19 | 0.4770 | 0.3080 | -0.250 | -1.184 |
| bc | LRU | 19 | 0.4924 | 0.3015 | -0.406 | -0.959 |
| bc | POPT | 19 | 0.4853 | 0.2997 | -0.412 | -1.092 |
| bc | SRRIP | 19 | 0.4748 | 0.2962 | -0.333 | -1.012 |
| bfs | GRASP | 19 | 0.6996 | 0.2994 | -1.282 | +0.756 |
| bfs | LRU | 19 | 0.6911 | 0.3413 | -1.132 | -0.128 |
| bfs | POPT | 19 | 0.6690 | 0.3129 | -1.174 | +0.446 |
| bfs | SRRIP | 19 | 0.6905 | 0.3372 | -1.135 | -0.052 |
| cc | GRASP | 16 | 0.3674 | 0.2483 | +0.304 | -0.871 |
| cc | LRU | 16 | 0.4419 | 0.2517 | -0.433 | -1.034 |
| cc | POPT | 16 | 0.4055 | 0.2581 | -0.019 | -1.310 |
| cc | SRRIP | 16 | 0.4154 | 0.2475 | -0.231 | -1.177 |
| pr | GRASP | 20 | 0.3486 | 0.3040 | +0.624 | -0.939 |
| pr | LRU | 20 | 0.4080 | 0.3275 | +0.375 | -1.289 |
| pr | POPT | 20 | 0.3221 | 0.2871 | +0.708 | -0.866 |
| pr | SRRIP | 20 | 0.3828 | 0.3224 | +0.504 | -1.137 |
| sssp | GRASP | 16 | 0.3451 | 0.3095 | +0.414 | -1.363 |
| sssp | LRU | 16 | 0.3284 | 0.2879 | +0.430 | -1.345 |
| sssp | POPT | 16 | 0.3083 | 0.2720 | +0.478 | -1.151 |
| sssp | SRRIP | 16 | 0.3175 | 0.2825 | +0.479 | -1.233 |

## Interpretation

- Skewness near 0 + negative excess kurtosis (platykurtic) means the distribution is light-tailed — the *opposite* of the pathological case (heavy tails / extreme outliers) that would bias percentile bootstrap CIs.
- Floor on |skewness| (2.0) and on |excess kurtosis| (7.0) come from Hesterberg 2015's published rules of thumb for bootstrap-CI applicability.
- Future regressions (corpus changes, scope changes) that push any (app, policy) cell beyond these floors will fail this gate and require switching to BCa / studentized bootstrap or reporting alternative CIs.

