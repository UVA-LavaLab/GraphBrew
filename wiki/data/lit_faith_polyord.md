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
| citation | hub | bc | 3 | -2.2269 | 1.0 | -3.6184 | 1.0 |
| citation | hub | bfs | 3 | -3.6266 | 1.0 | -4.4057 | 1.0 |
| citation | hub | cc | 3 | -6.5961 | 1.0 | -7.1836 | 1.0 |
| citation | hub | pr | 3 | -16.4988 | 1.0 | -10.1376 | 1.0 |
| citation | hub | sssp | 3 | -4.4013 | 1.0 | -4.3181 | 1.0 |
| mesh | no_hub | pr | 5 | -2.5389 | 0.8 | -0.2702 | 1.0 |
| road | no_hub | bc | 5 | -0.1559 | 0.8 | -0.0108 | 0.8 |
| road | no_hub | bfs | 5 | -0.0128 | 0.6 | -0.4173 | 0.8 |
| road | no_hub | cc | 5 | -0.152 | 1.0 | -1.2518 | 1.0 |
| road | no_hub | pr | 5 | -1.1678 | 0.8 | -0.0488 | 0.8 |
| road | no_hub | sssp | 5 | -0.0058 | 0.6 | -0.0603 | 0.6 |
| social | hub | bc | 12 | -3.5918 | 0.6667 | -4.2758 | 0.75 |
| social | hub | bfs | 12 | -2.9154 | 0.75 | -1.554 | 0.75 |
| social | hub | cc | 9 | -8.6674 | 0.8889 | -12.3912 | 0.8889 |
| social | hub | pr | 12 | -9.5867 | 0.75 | -5.5809 | 0.75 |
| social | hub | sssp | 9 | -3.4227 | 0.7778 | -2.9756 | 0.7778 |
| web | hub | bc | 3 | 2.729 | 0.0 | 2.107 | 0.0 |
| web | hub | bfs | 3 | -8.1601 | 1.0 | -2.9455 | 0.6667 |
| web | hub | cc | 3 | 0.0 | 0.3333 | 0.0 | 0.3333 |
| web | hub | pr | 3 | -2.719 | 1.0 | -1.3019 | 0.6667 |
| web | hub | sssp | 3 | 0.0 | 0.3333 | 2.4063 | 0.3333 |

## Per-app global (hub-bearing only)

| app | hub cells | POPT median (pp) | POPT improve frac |
|---|---|---|---|
| bc | 18 | -1.9215 | 0.6111 |
| bfs | 18 | -3.4801 | 0.8333 |
| cc | 15 | -8.4854 | 0.8 |
| pr | 18 | -10.8906 | 0.8333 |
| sssp | 15 | -3.4227 | 0.7333 |

_No violations._
