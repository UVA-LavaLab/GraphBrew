# Literature-faithfulness regime-sign audit (LIT-RegimeSign)

Per (graph_family, app, advice-policy) bucket in the per_observation table: hub families must not show majority regression and must keep their median delta below the hub ceiling; no-hub families may exhibit L-curve sign-flipping but their median must stay within the radius; no individual cell may exceed the extreme magnitude cap.

## Summary

| Metric | Value |
|---|---|
| Non-LRU per_observation rows | 342 |
| Buckets analysed | 72 |
| Hub buckets | 45 |
| No-hub buckets | 18 |
| Sign deadband | ±1.0 pp |
| Hub median ceiling | 0.5 pp |
| No-hub median radius | ±8.0 pp |
| Extreme magnitude cap | ±80.0 pp |
| Extreme cells (|Δ| > cap) | 0 |
| Violations | 0 |

## Per-bucket sign tally

| family | app | policy | n | neg | pos | zero | median Δ pp | max |Δ| pp |
|---|---|---|---:|---:|---:|---:|---:|---:|
| citation | bc | GRASP | 3 | 3 | 0 | 0 | -3.618 | 4.537 |
| citation | bc | POPT | 3 | 0 | 0 | 3 | -0.397 | 0.823 |
| citation | bc | SRRIP | 3 | 2 | 0 | 1 | -1.069 | 2.036 |
| citation | bfs | GRASP | 3 | 2 | 0 | 1 | -4.406 | 5.016 |
| citation | bfs | POPT | 3 | 2 | 0 | 1 | -2.643 | 6.283 |
| citation | bfs | SRRIP | 3 | 0 | 0 | 3 | -0.266 | 0.496 |
| citation | cc | GRASP | 3 | 3 | 0 | 0 | -6.729 | 7.064 |
| citation | cc | POPT | 3 | 3 | 0 | 0 | -3.951 | 8.859 |
| citation | cc | SRRIP | 3 | 3 | 0 | 0 | -2.332 | 4.058 |
| citation | pr | GRASP | 3 | 3 | 0 | 0 | -10.138 | 13.684 |
| citation | pr | POPT | 3 | 3 | 0 | 0 | -13.403 | 19.939 |
| citation | pr | SRRIP | 3 | 3 | 0 | 0 | -4.341 | 5.169 |
| citation | sssp | GRASP | 3 | 3 | 0 | 0 | -4.318 | 7.032 |
| citation | sssp | POPT | 3 | 2 | 1 | 0 | -1.406 | 1.754 |
| citation | sssp | SRRIP | 3 | 2 | 0 | 1 | -2.161 | 2.200 |
| mesh | pr | GRASP | 5 | 2 | 0 | 3 | -0.270 | 11.516 |
| mesh | pr | POPT | 5 | 3 | 0 | 2 | -2.221 | 15.636 |
| mesh | pr | SRRIP | 5 | 1 | 0 | 4 | +0.000 | 1.492 |
| road | bc | GRASP | 5 | 1 | 1 | 3 | -0.011 | 16.234 |
| road | bc | POPT | 5 | 1 | 1 | 3 | -0.130 | 17.674 |
| road | bc | SRRIP | 5 | 0 | 0 | 5 | +0.000 | 0.645 |
| road | bfs | GRASP | 5 | 2 | 1 | 2 | -0.417 | 62.586 |
| road | bfs | POPT | 5 | 1 | 1 | 3 | -0.006 | 35.126 |
| road | bfs | SRRIP | 5 | 0 | 1 | 4 | +0.000 | 4.381 |
| road | cc | GRASP | 5 | 1 | 0 | 4 | -0.202 | 4.657 |
| road | cc | POPT | 5 | 2 | 0 | 3 | -0.117 | 4.978 |
| road | cc | SRRIP | 5 | 0 | 0 | 5 | +0.000 | 0.083 |
| road | pr | GRASP | 5 | 0 | 1 | 4 | -0.049 | 2.552 |
| road | pr | POPT | 5 | 2 | 0 | 3 | -0.827 | 4.841 |
| road | pr | SRRIP | 5 | 0 | 0 | 5 | +0.000 | 0.054 |
| road | sssp | GRASP | 5 | 2 | 2 | 1 | -0.060 | 55.827 |
| road | sssp | POPT | 5 | 1 | 2 | 2 | -0.003 | 10.469 |
| road | sssp | SRRIP | 5 | 1 | 1 | 3 | -0.014 | 1.520 |
| small_world | bc | GRASP | 3 | 0 | 0 | 3 | +0.000 | 0.000 |
| small_world | bc | POPT | 3 | 0 | 0 | 3 | +0.000 | 0.000 |
| small_world | bc | SRRIP | 3 | 0 | 0 | 3 | +0.000 | 0.000 |
| small_world | bfs | GRASP | 3 | 0 | 0 | 3 | +0.000 | 0.000 |
| small_world | bfs | POPT | 3 | 0 | 0 | 3 | +0.000 | 0.000 |
| small_world | bfs | SRRIP | 3 | 0 | 0 | 3 | +0.000 | 0.000 |
| small_world | pr | GRASP | 3 | 0 | 0 | 3 | +0.000 | 0.000 |
| small_world | pr | POPT | 3 | 0 | 0 | 3 | +0.000 | 0.000 |
| small_world | pr | SRRIP | 3 | 0 | 0 | 3 | +0.000 | 0.000 |
| social | bc | GRASP | 9 | 9 | 0 | 0 | -4.848 | 7.212 |
| social | bc | POPT | 9 | 6 | 2 | 1 | -2.234 | 5.400 |
| social | bc | SRRIP | 9 | 9 | 0 | 0 | -2.617 | 3.362 |
| social | bfs | GRASP | 9 | 8 | 0 | 1 | -3.139 | 7.259 |
| social | bfs | POPT | 9 | 7 | 0 | 2 | -3.006 | 13.365 |
| social | bfs | SRRIP | 9 | 4 | 0 | 5 | -0.611 | 2.609 |
| social | cc | GRASP | 9 | 8 | 1 | 0 | -7.718 | 31.015 |
| social | cc | POPT | 9 | 8 | 0 | 1 | -6.478 | 11.045 |
| social | cc | SRRIP | 9 | 8 | 0 | 1 | -2.685 | 8.461 |
| social | pr | GRASP | 9 | 9 | 0 | 0 | -7.457 | 14.047 |
| social | pr | POPT | 9 | 9 | 0 | 0 | -8.430 | 12.901 |
| social | pr | SRRIP | 9 | 9 | 0 | 0 | -3.913 | 4.442 |
| social | sssp | GRASP | 9 | 7 | 0 | 2 | -2.976 | 11.387 |
| social | sssp | POPT | 9 | 4 | 3 | 2 | -0.051 | 4.288 |
| social | sssp | SRRIP | 9 | 7 | 0 | 2 | -2.015 | 3.534 |
| web | bc | GRASP | 3 | 0 | 2 | 1 | +2.107 | 4.246 |
| web | bc | POPT | 3 | 0 | 3 | 0 | +6.684 | 12.137 |
| web | bc | SRRIP | 3 | 1 | 0 | 2 | -0.961 | 1.729 |
| web | bfs | GRASP | 3 | 2 | 1 | 0 | -2.946 | 9.318 |
| web | bfs | POPT | 3 | 3 | 0 | 0 | -7.442 | 14.964 |
| web | bfs | SRRIP | 3 | 0 | 0 | 3 | -0.124 | 0.574 |
| web | cc | GRASP | 3 | 1 | 2 | 0 | +11.518 | 20.635 |
| web | cc | POPT | 3 | 1 | 0 | 2 | +0.000 | 5.454 |
| web | cc | SRRIP | 3 | 1 | 0 | 2 | +0.000 | 2.907 |
| web | pr | GRASP | 3 | 2 | 0 | 1 | -1.302 | 14.745 |
| web | pr | POPT | 3 | 2 | 0 | 1 | -1.632 | 16.850 |
| web | pr | SRRIP | 3 | 2 | 0 | 1 | -2.293 | 5.749 |
| web | sssp | GRASP | 3 | 1 | 2 | 0 | +2.406 | 5.814 |
| web | sssp | POPT | 3 | 0 | 1 | 2 | +0.000 | 2.281 |
| web | sssp | SRRIP | 3 | 0 | 0 | 3 | +0.000 | 0.760 |

_No violations._
