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
| citation | bc | GRASP | 3 | 2 | 1 | 0 | -2.062 | 4.959 |
| citation | bc | POPT | 3 | 2 | 0 | 1 | -1.544 | 3.617 |
| citation | bc | SRRIP | 3 | 1 | 0 | 2 | -0.908 | 1.796 |
| citation | bfs | GRASP | 3 | 3 | 0 | 0 | -4.229 | 8.678 |
| citation | bfs | POPT | 3 | 3 | 0 | 0 | -3.332 | 7.943 |
| citation | bfs | SRRIP | 3 | 0 | 0 | 3 | -0.409 | 0.489 |
| citation | cc | GRASP | 3 | 3 | 0 | 0 | -6.967 | 11.691 |
| citation | cc | POPT | 3 | 3 | 0 | 0 | -3.319 | 4.917 |
| citation | cc | SRRIP | 3 | 3 | 0 | 0 | -2.299 | 4.213 |
| citation | pr | GRASP | 3 | 3 | 0 | 0 | -11.149 | 13.543 |
| citation | pr | POPT | 3 | 3 | 0 | 0 | -14.260 | 22.195 |
| citation | pr | SRRIP | 3 | 3 | 0 | 0 | -4.277 | 5.045 |
| citation | sssp | GRASP | 3 | 3 | 0 | 0 | -6.230 | 8.068 |
| citation | sssp | POPT | 3 | 3 | 0 | 0 | -4.793 | 6.657 |
| citation | sssp | SRRIP | 3 | 2 | 0 | 1 | -2.301 | 2.406 |
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
| small_world | pr | GRASP | 3 | 0 | 0 | 3 | +0.000 | 0.000 |
| small_world | pr | POPT | 3 | 0 | 0 | 3 | +0.000 | 0.000 |
| small_world | pr | SRRIP | 3 | 0 | 0 | 3 | +0.000 | 0.000 |
| social | bc | GRASP | 9 | 9 | 0 | 0 | -5.844 | 8.662 |
| social | bc | POPT | 9 | 8 | 0 | 1 | -3.841 | 6.197 |
| social | bc | SRRIP | 9 | 9 | 0 | 0 | -2.541 | 3.324 |
| social | bfs | GRASP | 9 | 8 | 0 | 1 | -6.325 | 11.079 |
| social | bfs | POPT | 9 | 8 | 0 | 1 | -5.117 | 12.180 |
| social | bfs | SRRIP | 9 | 3 | 0 | 6 | -0.138 | 3.518 |
| social | cc | GRASP | 9 | 8 | 0 | 1 | -11.211 | 20.943 |
| social | cc | POPT | 9 | 7 | 0 | 2 | -3.290 | 14.756 |
| social | cc | SRRIP | 9 | 8 | 0 | 1 | -2.152 | 10.943 |
| social | pr | GRASP | 9 | 9 | 0 | 0 | -4.801 | 13.627 |
| social | pr | POPT | 9 | 9 | 0 | 0 | -9.610 | 13.714 |
| social | pr | SRRIP | 9 | 9 | 0 | 0 | -3.583 | 4.529 |
| social | sssp | GRASP | 9 | 7 | 0 | 2 | -3.080 | 10.823 |
| social | sssp | POPT | 9 | 7 | 0 | 2 | -3.722 | 7.014 |
| social | sssp | SRRIP | 9 | 7 | 0 | 2 | -2.045 | 3.990 |
| web | bc | GRASP | 3 | 2 | 0 | 1 | -1.343 | 2.652 |
| web | bc | POPT | 3 | 0 | 1 | 2 | +0.379 | 1.296 |
| web | bc | SRRIP | 3 | 2 | 0 | 1 | -1.163 | 1.335 |
| web | bfs | GRASP | 3 | 3 | 0 | 0 | -5.474 | 17.965 |
| web | bfs | POPT | 3 | 3 | 0 | 0 | -7.982 | 18.153 |
| web | bfs | SRRIP | 3 | 0 | 0 | 3 | -0.233 | 0.777 |
| web | cc | GRASP | 3 | 1 | 0 | 2 | -0.005 | 6.636 |
| web | cc | POPT | 3 | 1 | 0 | 2 | -0.002 | 5.369 |
| web | cc | SRRIP | 3 | 1 | 0 | 2 | -0.000 | 3.859 |
| web | pr | GRASP | 3 | 3 | 0 | 0 | -3.574 | 14.770 |
| web | pr | POPT | 3 | 3 | 0 | 0 | -2.511 | 18.400 |
| web | pr | SRRIP | 3 | 2 | 0 | 1 | -2.244 | 5.668 |
| web | sssp | GRASP | 3 | 0 | 0 | 3 | +0.001 | 0.034 |
| web | sssp | POPT | 3 | 0 | 0 | 3 | -0.003 | 0.824 |
| web | sssp | SRRIP | 3 | 0 | 0 | 3 | -0.003 | 0.502 |

_No violations._
