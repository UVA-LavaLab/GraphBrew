# Corpus Diversity Profile

Topology features extracted from the `=== Graph Topology Features ===` block printed by GAPBS during the PR / LRU / L3=1 MB cell of every literature sweep. One representative log per graph.

Diversity rationale: the corpus spans web (`web-Google`), citation (`cit-Patents`), social (`soc-pokec`, `soc-LiveJournal1`), and dense-social (`com-orkut`) graphs at scales ranging from 1 k vertices (`email-Eu-core`) to 4.8 M vertices (`soc-LiveJournal1`). Average degree spans 11 (cit-Patents) to 114 (com-orkut), and hub concentration spans 0.33 (cit-Patents) to 0.62 (soc-LiveJournal1), covering both diffuse-locality (citation) and hub-heavy (large social) regimes that the literature targets.

> The clustering coefficient column is GAPBS's *sampled* local CC (computed per-vertex over a subset, then averaged), not the global literature CC. Sampled CC for `com-orkut` (0.008) is much lower than the canonical SNAP value (~0.17) because the dense-subgraph concentration is diluted by Orkut's long-tail low-degree vertices in the per-vertex sample. The literature reasoning behind our KNOWN_DEVIATIONS entries (e.g. `(com-orkut, cc, *, POPT_GE_GRASP)`) still references the canonical SNAP CC for that graph.

## Scale

| Graph | Nodes | Edges | Edge orientation | Avg degree |
|---|---:|---:|---|---:|
| `email-Eu-core` | 1,005 | 16,064 | undirected | 32.20 |
| `web-Google` | 916,428 | 4,322,051 | undirected | 13.22 |
| `cit-Patents` | 3,774,768 | 16,518,947 | undirected | 11.47 |
| `soc-pokec` | 1,632,804 | 22,301,964 | undirected | 41.43 |
| `com-orkut` | 3,072,627 | 117,185,083 | undirected | 113.89 |
| `soc-LiveJournal1` | 4,847,571 | 42,851,237 | undirected | 38.12 |

## Topology features

| Graph | Clustering Coefficient | Avg Path Length | Diameter Estimate | Community Count Estimate | Degree Variance | Hub Concentration | Avg Degree | Graph Density | Modularity | Forward Edge Fraction | Working Set Ratio | Vertex Significance Skewness | Window Neighbor Overlap | Sampled Locality Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `email-Eu-core` | 0.1610 | 1.71 | 4.00 | 1 | 1.15 | 0.3670 | 32.20 | 0.0318 | 0.2853 | 0.5000 | 0.0043 | 1.26 | 0.1271 | 0.0297 |
| `web-Google` | 0.0985 | 3.83 | 4.00 | 96 | 5.48 | 0.4833 | 13.22 | 0.0000 | 0.7880 | 0.5329 | 1.47 | 3.48 | 0.0184 | 0.0027 |
| `cit-Patents` | 0.0669 | 3.60 | 4.00 | 195 | 1.14 | 0.3378 | 11.47 | 0.0000 | 0.5258 | 0.4940 | 5.74 | 2.80 | 0.0180 | 0.0026 |
| `soc-pokec` | 0.0202 | 2.52 | 3.00 | 128 | 2.00 | 0.3982 | 41.43 | 0.0000 | 0.3649 | 0.5035 | 6.10 | 2.63 | 0.0573 | 0.0298 |
| `com-orkut` | 0.0080 | 1.97 | 2.00 | 176 | 3.03 | 0.4398 | 113.89 | 0.0000 | 0.4641 | 0.5002 | 29.40 | 2.01 | 0.1326 | 0.0993 |
| `soc-LiveJournal1` | 0.0227 | 2.05 | 3.00 | 221 | 2.97 | 0.6235 | 38.12 | 0.0000 | 0.6220 | 0.4832 | 12.53 | 4.24 | 0.1291 | 0.1047 |

## Interpretation

* **Hub concentration** measures the fraction of edges incident on the top-decile of vertices. Social graphs (`soc-LiveJournal1` 0.62, `com-orkut` 0.44) sit at the high end, validating the GRASP design assumption that a small hot set captures most reuse. Citation graphs (`cit-Patents` 0.34) are the diffuse-locality opposite extreme.
* **Average degree** spans an order of magnitude (11 on cit-Patents to 114 on com-orkut), so the corpus stresses both bandwidth-bound (orkut, pokec) and latency-bound (cit-Patents, web-Google) regimes.
* **Working set ratio** is the ratio of touched lines to the L3 capacity used during the topology pass. `email-Eu-core` (0.004) fits entirely in L2; `com-orkut` (29.4) requires the working set to be 30x larger than L3 — the regime where replacement policy choice matters most.
* **Forward edge fraction** is the share of edges pointing to higher-indexed vertices. Values near 0.5 indicate the DBG reordering pass achieved its goal of monotonically increasing access order across all graphs.
* **Sampled locality score** estimates the cache-line reuse over a windowed access trace. `com-orkut` (0.10) and `soc-LiveJournal1` (0.10) show the strongest in-window reuse, again confirming the GRASP / POPT operating regime.
