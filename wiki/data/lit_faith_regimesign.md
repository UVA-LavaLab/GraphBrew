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
| citation | bc | GRASP | 3 | 2 | 0 | 1 | -2.053 | 5.271 |
| citation | bc | POPT | 3 | 1 | 0 | 2 | -0.456 | 2.742 |
| citation | bc | SRRIP | 3 | 2 | 0 | 1 | -1.203 | 2.393 |
| citation | bfs | GRASP | 3 | 2 | 0 | 1 | -4.081 | 8.624 |
| citation | bfs | POPT | 3 | 2 | 0 | 1 | -3.701 | 7.936 |
| citation | bfs | SRRIP | 3 | 0 | 0 | 3 | -0.397 | 0.685 |
| citation | cc | GRASP | 3 | 3 | 0 | 0 | -6.967 | 11.691 |
| citation | cc | POPT | 3 | 3 | 0 | 0 | -3.319 | 4.917 |
| citation | cc | SRRIP | 3 | 3 | 0 | 0 | -2.299 | 4.213 |
| citation | pr | GRASP | 3 | 3 | 0 | 0 | -11.025 | 13.369 |
| citation | pr | POPT | 3 | 3 | 0 | 0 | -13.859 | 21.615 |
| citation | pr | SRRIP | 3 | 3 | 0 | 0 | -4.181 | 4.915 |
| citation | sssp | GRASP | 3 | 3 | 0 | 0 | -6.904 | 8.673 |
| citation | sssp | POPT | 3 | 3 | 0 | 0 | -5.271 | 6.869 |
| citation | sssp | SRRIP | 3 | 2 | 0 | 1 | -2.465 | 2.498 |
| mesh | pr | GRASP | 5 | 3 | 0 | 2 | -1.416 | 13.730 |
| mesh | pr | POPT | 5 | 3 | 0 | 2 | -1.650 | 17.597 |
| mesh | pr | SRRIP | 5 | 1 | 0 | 4 | -0.007 | 1.455 |
| road | bc | GRASP | 5 | 0 | 2 | 3 | -0.053 | 36.032 |
| road | bc | POPT | 5 | 1 | 1 | 3 | -0.320 | 20.444 |
| road | bc | SRRIP | 5 | 2 | 0 | 3 | -0.053 | 2.097 |
| road | bfs | GRASP | 5 | 0 | 2 | 3 | -0.052 | 68.404 |
| road | bfs | POPT | 5 | 2 | 1 | 2 | -0.068 | 33.391 |
| road | bfs | SRRIP | 5 | 1 | 1 | 3 | -0.001 | 4.643 |
| road | cc | GRASP | 5 | 0 | 2 | 3 | -0.050 | 9.120 |
| road | cc | POPT | 5 | 0 | 1 | 4 | +0.010 | 5.058 |
| road | cc | SRRIP | 5 | 0 | 0 | 5 | -0.043 | 0.537 |
| road | pr | GRASP | 5 | 0 | 1 | 4 | -0.022 | 1.472 |
| road | pr | POPT | 5 | 3 | 0 | 2 | -1.660 | 5.228 |
| road | pr | SRRIP | 5 | 0 | 0 | 5 | -0.000 | 0.237 |
| road | sssp | GRASP | 5 | 0 | 3 | 2 | +5.401 | 70.117 |
| road | sssp | POPT | 5 | 2 | 2 | 1 | -0.151 | 13.820 |
| road | sssp | SRRIP | 5 | 1 | 2 | 2 | +0.000 | 2.756 |
| small_world | bc | GRASP | 3 | 0 | 0 | 3 | +0.000 | 0.000 |
| small_world | bc | POPT | 3 | 0 | 0 | 3 | +0.000 | 0.010 |
| small_world | bc | SRRIP | 3 | 0 | 0 | 3 | -0.000 | 0.000 |
| small_world | bfs | GRASP | 3 | 0 | 1 | 2 | -0.000 | 17.752 |
| small_world | bfs | POPT | 3 | 0 | 0 | 3 | -0.000 | 0.003 |
| small_world | bfs | SRRIP | 3 | 0 | 0 | 3 | -0.001 | 0.002 |
| small_world | pr | GRASP | 3 | 0 | 0 | 3 | -0.013 | 0.329 |
| small_world | pr | POPT | 3 | 0 | 0 | 3 | -0.021 | 0.059 |
| small_world | pr | SRRIP | 3 | 0 | 0 | 3 | -0.024 | 0.039 |
| social | bc | GRASP | 9 | 9 | 0 | 0 | -5.844 | 8.691 |
| social | bc | POPT | 9 | 8 | 0 | 1 | -3.770 | 6.331 |
| social | bc | SRRIP | 9 | 9 | 0 | 0 | -2.541 | 3.397 |
| social | bfs | GRASP | 9 | 7 | 0 | 2 | -2.603 | 11.079 |
| social | bfs | POPT | 9 | 8 | 0 | 1 | -2.709 | 12.180 |
| social | bfs | SRRIP | 9 | 1 | 0 | 8 | -0.138 | 1.581 |
| social | cc | GRASP | 9 | 8 | 0 | 1 | -11.211 | 20.943 |
| social | cc | POPT | 9 | 7 | 0 | 2 | -3.290 | 14.756 |
| social | cc | SRRIP | 9 | 8 | 0 | 1 | -2.152 | 10.943 |
| social | pr | GRASP | 9 | 9 | 0 | 0 | -4.791 | 13.617 |
| social | pr | POPT | 9 | 9 | 0 | 0 | -9.508 | 13.446 |
| social | pr | SRRIP | 9 | 9 | 0 | 0 | -3.534 | 4.517 |
| social | sssp | GRASP | 9 | 6 | 1 | 2 | -2.657 | 11.039 |
| social | sssp | POPT | 9 | 7 | 0 | 2 | -2.245 | 6.942 |
| social | sssp | SRRIP | 9 | 6 | 0 | 3 | -1.335 | 3.990 |
| web | bc | GRASP | 3 | 2 | 0 | 1 | -1.583 | 2.691 |
| web | bc | POPT | 3 | 0 | 1 | 2 | +0.211 | 1.467 |
| web | bc | SRRIP | 3 | 2 | 0 | 1 | -1.128 | 1.838 |
| web | bfs | GRASP | 3 | 3 | 0 | 0 | -5.442 | 17.683 |
| web | bfs | POPT | 3 | 3 | 0 | 0 | -7.923 | 18.069 |
| web | bfs | SRRIP | 3 | 0 | 0 | 3 | -0.102 | 0.481 |
| web | cc | GRASP | 3 | 1 | 0 | 2 | -0.005 | 6.636 |
| web | cc | POPT | 3 | 1 | 0 | 2 | -0.002 | 5.369 |
| web | cc | SRRIP | 3 | 1 | 0 | 2 | -0.000 | 3.859 |
| web | pr | GRASP | 3 | 3 | 0 | 0 | -3.583 | 14.766 |
| web | pr | POPT | 3 | 3 | 0 | 0 | -2.510 | 18.413 |
| web | pr | SRRIP | 3 | 2 | 0 | 1 | -2.238 | 5.698 |
| web | sssp | GRASP | 3 | 0 | 0 | 3 | +0.001 | 0.034 |
| web | sssp | POPT | 3 | 0 | 0 | 3 | -0.003 | 0.824 |
| web | sssp | SRRIP | 3 | 0 | 0 | 3 | -0.003 | 0.502 |

_No violations._
