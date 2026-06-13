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
| Per-app global POPT improve-frac floor (hub) | ≥ 0.45 |
| Cell-count floor (hub) | ≥ 2 |
| Violations | 0 |

## Per-bucket detail

| family | regime | app | n | POPT med (pp) | POPT improve frac | GRASP med (pp) | GRASP improve frac |
|---|---|---|---|---|---|---|---|
| citation | hub | bc | 3 | -0.3972 | 0.6667 | -3.6184 | 1.0 |
| citation | hub | bfs | 3 | -2.6434 | 1.0 | -4.4057 | 1.0 |
| citation | hub | cc | 3 | -3.9514 | 1.0 | -6.729 | 1.0 |
| citation | hub | pr | 3 | -13.4031 | 1.0 | -10.1376 | 1.0 |
| citation | hub | sssp | 3 | -1.4064 | 0.6667 | -4.3181 | 1.0 |
| mesh | no_hub | pr | 5 | -2.2207 | 0.8 | -0.2702 | 1.0 |
| road | no_hub | bc | 5 | -0.1296 | 0.8 | -0.0108 | 0.8 |
| road | no_hub | bfs | 5 | -0.0065 | 0.6 | -0.4173 | 0.8 |
| road | no_hub | cc | 5 | -0.1171 | 1.0 | -0.2023 | 1.0 |
| road | no_hub | pr | 5 | -0.8272 | 0.8 | -0.0488 | 0.8 |
| road | no_hub | sssp | 5 | -0.0032 | 0.6 | -0.0603 | 0.6 |
| social | hub | bc | 12 | -0.8314 | 0.5833 | -4.2758 | 0.75 |
| social | hub | bfs | 12 | -1.9394 | 0.75 | -1.554 | 0.75 |
| social | hub | cc | 9 | -6.4783 | 0.8889 | -7.7183 | 0.8889 |
| social | hub | pr | 12 | -6.7819 | 0.75 | -5.5809 | 0.75 |
| social | hub | sssp | 9 | -0.051 | 0.5556 | -2.9756 | 0.7778 |
| web | hub | bc | 3 | 6.6835 | 0.0 | 2.107 | 0.0 |
| web | hub | bfs | 3 | -7.4418 | 1.0 | -2.9455 | 0.6667 |
| web | hub | cc | 3 | 0.0 | 0.3333 | 11.5175 | 0.3333 |
| web | hub | pr | 3 | -1.6317 | 1.0 | -1.3019 | 0.6667 |
| web | hub | sssp | 3 | 0.0 | 0.0 | 2.4063 | 0.3333 |

## Per-app global (hub-bearing only)

| app | hub cells | POPT median (pp) | POPT improve frac |
|---|---|---|---|
| bc | 18 | -0.1455 | 0.5 |
| bfs | 18 | -2.2968 | 0.8333 |
| cc | 15 | -5.4535 | 0.8 |
| pr | 18 | -7.6542 | 0.8333 |
| sssp | 15 | 0.0 | 0.4667 |

_No violations._
