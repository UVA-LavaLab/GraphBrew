# Literature-faithfulness policy-ordering audit (LIT-PolyOrd)

Per (graph_family × app) bucket: locks the direction of the POPT-vs-LRU and GRASP-vs-LRU orderings on hub-bearing families (social / citation / web) while letting hub-less families (road / mesh) regress as the literature documents.

## Summary

| Metric | Value |
|---|---|
| Bucket count | 21 |
| Hub-bearing buckets | 15 |
| Hub-less buckets | 6 |
| POPT median bound (hub) | ≤ 0.5 pp |
| GRASP median bound (hub) | ≤ 1.0 pp |
| POPT improve-frac floor (hub, n≥5) | ≥ 0.5 |
| Per-app global POPT improve-frac floor (hub) | ≥ 0.55 |
| Cell-count floor (hub) | ≥ 2 |
| Violations | 0 |

## Per-bucket detail

| family | regime | app | n | POPT med (pp) | POPT improve frac | GRASP med (pp) | GRASP improve frac |
|---|---|---|---|---|---|---|---|
| citation | hub | bc | 3 | -1.5437 | 1.0 | -2.0618 | 0.6667 |
| citation | hub | bfs | 3 | -3.3319 | 1.0 | -4.2292 | 1.0 |
| citation | hub | cc | 3 | -3.319 | 1.0 | -6.9669 | 1.0 |
| citation | hub | pr | 3 | -14.2599 | 1.0 | -11.1489 | 1.0 |
| citation | hub | sssp | 3 | -4.793 | 1.0 | -6.2298 | 1.0 |
| mesh | no_hub | pr | 5 | -1.6501 | 1.0 | -1.4162 | 1.0 |
| road | no_hub | bc | 5 | -0.3195 | 0.8 | -0.0534 | 0.6 |
| road | no_hub | bfs | 5 | -0.0677 | 0.8 | -0.052 | 0.6 |
| road | no_hub | cc | 5 | 0.0101 | 0.4 | -0.0499 | 0.6 |
| road | no_hub | pr | 5 | -1.66 | 1.0 | -0.0221 | 0.6 |
| road | no_hub | sssp | 5 | -0.1514 | 0.6 | 5.4007 | 0.4 |
| social | hub | bc | 12 | -3.3808 | 0.75 | -4.953 | 0.8333 |
| social | hub | bfs | 12 | -2.9936 | 0.8333 | -3.0435 | 0.9167 |
| social | hub | cc | 9 | -3.2899 | 0.8889 | -11.2108 | 1.0 |
| social | hub | pr | 12 | -8.3566 | 0.75 | -3.9205 | 0.75 |
| social | hub | sssp | 9 | -3.722 | 1.0 | -3.0803 | 1.0 |
| web | hub | bc | 3 | 0.3785 | 0.3333 | -1.3433 | 0.6667 |
| web | hub | bfs | 3 | -7.9818 | 1.0 | -5.4739 | 1.0 |
| web | hub | cc | 3 | -0.0019 | 0.6667 | -0.0054 | 0.6667 |
| web | hub | pr | 3 | -2.5114 | 1.0 | -3.5741 | 1.0 |
| web | hub | sssp | 3 | -0.0028 | 0.6667 | 0.0011 | 0.3333 |

## Per-app global (hub-bearing only)

| app | hub cells | POPT median (pp) | POPT improve frac |
|---|---|---|---|
| bc | 18 | -1.3601 | 0.7222 |
| bfs | 18 | -3.6379 | 0.8889 |
| cc | 15 | -3.2899 | 0.8667 |
| pr | 18 | -9.4944 | 0.8333 |
| sssp | 15 | -2.7818 | 0.9333 |

_No violations._
