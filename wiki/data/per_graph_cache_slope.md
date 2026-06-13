# Per-graph oracle-gap cache-sensitivity slope

Source: `wiki/data/oracle_gap.json`  •  paper L3: 1MB, 4MB, 8MB  •  graphs with full trajectory: 6  •  full (graph, app, policy) trajectories: 112

Significant anti-scaling threshold: **+1.0 pp** per octave. A cell that grows its gap by that much (or more) at any single octave is flagged.

## Headline

- Cells with significant anti-scaling: **48 / 112**
- Of those, **22** belong to the oracle-aware policies (GRASP + POPT). The remainder are LRU + SRRIP.

## Per-policy anti-scaling cell count

| policy | n_cells_with_anti_scaling |
|---|---:|
| **GRASP** | 10 |
| **LRU** | 15 |
| **POPT** | 12 |
| **SRRIP** | 11 |

## Per-graph anti-scaling cell count

| graph | family | n_cells_with_anti_scaling |
|---|---|---:|
| cit-Patents | citation | 13 |
| com-orkut | social | 8 |
| email-Eu-core | social | 0 |
| soc-LiveJournal1 | social | 12 |
| soc-pokec | social | 7 |
| web-Google | web | 8 |

## Top anti-scaling cells (largest single-octave gap growth)

| graph | family | app | policy | max octave growth | 1MB → 4MB → 8MB gap |
|---|---|---|---|---:|---|
| soc-pokec | social | cc | GRASP | +26.70 pp | 0.00 → 4.32 → 31.02 |
| web-Google | web | cc | GRASP | +19.68 pp | 0.96 → 20.64 → 11.52 |
| web-Google | web | bfs | GRASP | +12.02 pp | 0.00 → 12.02 → 16.76 |
| web-Google | web | bfs | LRU | +11.36 pp | 3.60 → 14.96 → 7.44 |
| web-Google | web | bfs | SRRIP | +11.32 pp | 3.52 → 14.84 → 6.87 |
| com-orkut | social | cc | SRRIP | +9.56 pp | 3.71 → 13.27 → 2.62 |
| soc-pokec | social | bfs | GRASP | +8.80 pp | 0.00 → 2.99 → 11.79 |
| com-orkut | social | cc | POPT | +8.38 pp | 4.51 → 12.89 → 0.00 |
| com-orkut | social | cc | LRU | +7.66 pp | 7.06 → 14.71 → 11.04 |
| soc-pokec | social | bfs | SRRIP | +7.25 pp | 3.29 → 5.52 → 12.76 |
| soc-pokec | social | bfs | LRU | +7.24 pp | 4.69 → 6.13 → 13.37 |
| cit-Patents | citation | pr | LRU | +6.54 pp | 13.40 → 19.94 → 12.04 |
| web-Google | web | sssp | GRASP | +5.81 pp | 0.00 → 5.81 → 2.41 |
| web-Google | web | bc | POPT | +4.90 pp | 3.51 → 8.41 → 13.10 |
| com-orkut | social | pr | GRASP | +3.84 pp | 0.00 → 3.84 → 2.73 |
| com-orkut | social | bc | POPT | +3.82 pp | 2.68 → 6.50 → 8.95 |
| cit-Patents | citation | pr | SRRIP | +3.68 pp | 11.92 → 15.60 → 6.87 |
| cit-Patents | citation | bfs | LRU | +3.64 pp | 0.76 → 4.41 → 6.28 |
| cit-Patents | citation | bfs | SRRIP | +3.39 pp | 0.52 → 3.91 → 6.02 |
| web-Google | web | bc | GRASP | +3.23 pp | 2.11 → 1.97 → 5.21 |
| cit-Patents | citation | sssp | POPT | +3.06 pp | 2.56 → 5.62 → 5.78 |
| soc-LiveJournal1 | social | pr | LRU | +2.79 pp | 10.12 → 12.90 → 9.51 |
| cit-Patents | citation | cc | POPT | +2.78 pp | 0.00 → 2.78 → 1.16 |
| cit-Patents | citation | sssp | LRU | +2.71 pp | 4.32 → 7.03 → 4.26 |
| soc-pokec | social | bc | POPT | +2.51 pp | 2.16 → 4.68 → 4.21 |

## Interpretation

- If `n_oracle_aware_anti_scaling` is 0, the paper can claim: 'no individual graph shows GRASP or POPT regressing as L3 grows' — strictly stronger than the corpus-averaged finding in gate 52.
- If a small handful of (graph, app) cells flag GRASP/POPT anti-scaling, those should be called out specifically in the paper as known exceptions worth disclosing.
