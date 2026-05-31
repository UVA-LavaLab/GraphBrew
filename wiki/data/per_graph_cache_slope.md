# Per-graph oracle-gap cache-sensitivity slope

Source: `wiki/data/oracle_gap.json`  •  paper L3: 1MB, 4MB, 8MB  •  graphs with full trajectory: 6  •  full (graph, app, policy) trajectories: 112

Significant anti-scaling threshold: **+1.0 pp** per octave. A cell that grows its gap by that much (or more) at any single octave is flagged.

## Headline

- Cells with significant anti-scaling: **33 / 112**
- Of those, **8** belong to the oracle-aware policies (GRASP + POPT). The remainder are LRU + SRRIP.

## Per-policy anti-scaling cell count

| policy | n_cells_with_anti_scaling |
|---|---:|
| **GRASP** | 2 |
| **LRU** | 13 |
| **POPT** | 6 |
| **SRRIP** | 12 |

## Per-graph anti-scaling cell count

| graph | family | n_cells_with_anti_scaling |
|---|---|---:|
| cit-Patents | citation | 10 |
| com-orkut | social | 5 |
| email-Eu-core | social | 0 |
| soc-LiveJournal1 | social | 8 |
| soc-pokec | social | 3 |
| web-Google | web | 7 |

## Top anti-scaling cells (largest single-octave gap growth)

| graph | family | app | policy | max octave growth | 1MB → 4MB → 8MB gap |
|---|---|---|---|---:|---|
| web-Google | web | bfs | LRU | +14.73 pp | 3.42 → 18.15 → 7.98 |
| web-Google | web | bfs | SRRIP | +14.56 pp | 3.36 → 17.92 → 7.21 |
| cit-Patents | citation | pr | LRU | +9.88 pp | 12.31 → 22.20 → 14.26 |
| com-orkut | social | cc | LRU | +7.60 pp | 10.52 → 13.35 → 20.94 |
| cit-Patents | citation | pr | GRASP | +7.23 pp | 1.42 → 8.65 → 3.11 |
| cit-Patents | citation | pr | SRRIP | +7.07 pp | 10.85 → 17.92 → 9.21 |
| soc-pokec | social | bfs | SRRIP | +5.57 pp | 1.90 → 6.51 → 12.08 |
| soc-pokec | social | bfs | LRU | +5.53 pp | 2.87 → 6.65 → 12.18 |
| com-orkut | social | cc | SRRIP | +5.17 pp | 6.73 → 11.90 → 10.00 |
| cit-Patents | citation | sssp | LRU | +5.01 pp | 3.06 → 8.07 → 6.23 |
| cit-Patents | citation | bfs | SRRIP | +4.68 pp | 0.74 → 3.74 → 8.42 |
| soc-LiveJournal1 | social | cc | LRU | +4.58 pp | 3.82 → 6.63 → 11.21 |
| cit-Patents | citation | bfs | LRU | +4.45 pp | 1.15 → 4.23 → 8.68 |
| soc-pokec | social | cc | LRU | +3.86 pp | 13.11 → 16.97 → 0.04 |
| soc-LiveJournal1 | social | cc | SRRIP | +3.56 pp | 2.72 → 5.50 → 9.06 |
| cit-Patents | citation | sssp | SRRIP | +3.02 pp | 2.64 → 5.66 → 3.93 |
| cit-Patents | citation | bc | LRU | +2.90 pp | 0.48 → 2.06 → 4.96 |
| soc-LiveJournal1 | social | pr | LRU | +2.78 pp | 10.93 → 13.71 → 9.74 |
| soc-LiveJournal1 | social | pr | SRRIP | +2.48 pp | 7.60 → 10.08 → 6.16 |
| web-Google | web | bfs | GRASP | +2.32 pp | 0.00 → 0.19 → 2.51 |
| com-orkut | social | bc | POPT | +2.19 pp | 2.48 → 4.67 → 4.73 |
| soc-LiveJournal1 | social | sssp | SRRIP | +2.14 pp | 2.16 → 4.30 → 2.42 |
| cit-Patents | citation | bc | SRRIP | +2.01 pp | 0.00 → 1.15 → 3.16 |
| com-orkut | social | bfs | LRU | +1.90 pp | 0.23 → 2.04 → 3.94 |
| soc-LiveJournal1 | social | sssp | LRU | +1.89 pp | 4.76 → 6.64 → 4.47 |

## Interpretation

- If `n_oracle_aware_anti_scaling` is 0, the paper can claim: 'no individual graph shows GRASP or POPT regressing as L3 grows' — strictly stronger than the corpus-averaged finding in gate 52.
- If a small handful of (graph, app) cells flag GRASP/POPT anti-scaling, those should be called out specifically in the paper as known exceptions worth disclosing.
