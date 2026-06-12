# Per-graph oracle-gap cache-sensitivity slope

Source: `wiki/data/oracle_gap.json`  •  paper L3: 1MB, 4MB, 8MB  •  graphs with full trajectory: 6  •  full (graph, app, policy) trajectories: 112

Significant anti-scaling threshold: **+1.0 pp** per octave. A cell that grows its gap by that much (or more) at any single octave is flagged.

## Headline

- Cells with significant anti-scaling: **44 / 112**
- Of those, **16** belong to the oracle-aware policies (GRASP + POPT). The remainder are LRU + SRRIP.

## Per-policy anti-scaling cell count

| policy | n_cells_with_anti_scaling |
|---|---:|
| **GRASP** | 8 |
| **LRU** | 16 |
| **POPT** | 8 |
| **SRRIP** | 12 |

## Per-graph anti-scaling cell count

| graph | family | n_cells_with_anti_scaling |
|---|---|---:|
| cit-Patents | citation | 12 |
| com-orkut | social | 8 |
| email-Eu-core | social | 0 |
| soc-LiveJournal1 | social | 10 |
| soc-pokec | social | 7 |
| web-Google | web | 7 |

## Top anti-scaling cells (largest single-octave gap growth)

| graph | family | app | policy | max octave growth | 1MB → 4MB → 8MB gap |
|---|---|---|---|---:|---|
| web-Google | web | bfs | GRASP | +15.48 pp | 0.00 → 15.48 → 17.48 |
| web-Google | web | bfs | LRU | +14.82 pp | 3.60 → 18.43 → 8.16 |
| web-Google | web | bfs | SRRIP | +14.78 pp | 3.52 → 18.30 → 7.59 |
| com-orkut | social | cc | LRU | +9.99 pp | 10.17 → 15.70 → 25.69 |
| soc-pokec | social | cc | LRU | +8.94 pp | 12.39 → 21.33 → 0.00 |
| cit-Patents | citation | pr | LRU | +7.76 pp | 16.50 → 24.26 → 14.74 |
| com-orkut | social | cc | SRRIP | +7.44 pp | 6.82 → 14.26 → 17.26 |
| com-orkut | social | pr | GRASP | +7.22 pp | 0.30 → 7.52 → 4.02 |
| soc-pokec | social | bfs | GRASP | +7.04 pp | 0.00 → 4.78 → 11.82 |
| web-Google | web | sssp | GRASP | +5.81 pp | 0.00 → 5.81 → 2.41 |
| com-orkut | social | cc | POPT | +5.77 pp | 5.94 → 11.72 → 8.98 |
| soc-pokec | social | bfs | SRRIP | +5.49 pp | 3.29 → 7.31 → 12.79 |
| soc-pokec | social | bfs | LRU | +5.48 pp | 4.69 → 7.92 → 13.40 |
| cit-Patents | citation | pr | SRRIP | +4.91 pp | 15.02 → 19.92 → 9.57 |
| soc-LiveJournal1 | social | cc | LRU | +4.65 pp | 11.67 → 10.48 → 15.13 |
| cit-Patents | citation | bfs | SRRIP | +3.93 pp | 0.67 → 3.91 → 7.84 |
| soc-LiveJournal1 | social | cc | POPT | +3.81 pp | 0.00 → 1.99 → 5.81 |
| cit-Patents | citation | bfs | LRU | +3.70 pp | 0.91 → 4.41 → 8.11 |
| soc-LiveJournal1 | social | cc | SRRIP | +3.52 pp | 9.26 → 8.28 → 11.80 |
| web-Google | web | bc | POPT | +3.32 pp | 1.89 → 4.46 → 7.78 |
| web-Google | web | bc | GRASP | +3.23 pp | 2.11 → 1.97 → 5.21 |
| soc-pokec | social | cc | SRRIP | +3.16 pp | 9.71 → 12.87 → 0.00 |
| cit-Patents | citation | bfs | GRASP | +3.09 pp | 0.15 → 0.00 → 3.09 |
| com-orkut | social | bc | POPT | +2.83 pp | 0.84 → 3.67 → 5.80 |
| cit-Patents | citation | sssp | LRU | +2.71 pp | 4.32 → 7.03 → 4.40 |

## Interpretation

- If `n_oracle_aware_anti_scaling` is 0, the paper can claim: 'no individual graph shows GRASP or POPT regressing as L3 grows' — strictly stronger than the corpus-averaged finding in gate 52.
- If a small handful of (graph, app) cells flag GRASP/POPT anti-scaling, those should be called out specifically in the paper as known exceptions worth disclosing.
